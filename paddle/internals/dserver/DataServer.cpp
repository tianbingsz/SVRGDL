/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#include <iostream>

#include "DataServer.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Util.h"

P_DECLARE_int32(num_gradient_servers);
P_DECLARE_string(pservers);
P_DECLARE_int32(ports_num);

namespace paddle {

DataServer::DataServer(const std::string& addr,
                       int port,
                       int rdmaCpu)
    : ProtoServer(addr, port, rdmaCpu), serverId_(-1) {
  REGISTER_SERVICE_FUNCTION_EX(DataServer, sendData);
  REGISTER_SERVICE_FUNCTION(DataServer, synchronize);
  serviceNum_ = calculateServiceNum(FLAGS_pservers, FLAGS_ports_num);
  CHECK_GT(serviceNum_, 0U);
}

bool DataServer::init() {
  refMems_.resize(FLAGS_num_gradient_servers);
  refLabelMems_.resize(FLAGS_num_gradient_servers);
  refGradMems_.resize(FLAGS_num_gradient_servers);
  synchronizeBarriers_.resize(SyncObject_ARRAYSIZE);
  for (auto& barrier : synchronizeBarriers_) {
    barrier.reset(new ThreadBarrier(FLAGS_num_gradient_servers));
  }
  return true;
}

void DataServer::setData(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback,
                         std::vector<CpuMemHandlePtr>& mem) {
  SendDataResponse response;
  int64_t serverId = request.server_id();
  if (serverId_ < 0) {
    serverId_ = serverId;
  } else {
    CHECK_EQ(serverId_, serverId);
  }
  response.set_type(request.type());
  response.set_server_id(request.server_id());
  /* each trainer sends its output features to dataServers,
   * Features will be divided into server number blocks.
   * each dataServer gets one block. In order to identify
   * which trainer does the block come from, put the block in its
   * corresponding position. For example, block A comes form trainer
   * whoes trainer_id is i, then put block A in refMems_[i].*/
  CHECK_EQ(msgReader->getNumBlocks(), (size_t)(request.blocks_size()));
  size_t totalLen = msgReader->getTotalLength();
  VLOG(2) << "Total len: " << totalLen;
  if (totalLen > 0) {
    CHECK_EQ(msgReader->getNumBlocks(), 1U)
        << "Only one block currently support now!";
    auto& block = request.blocks(0);
    int64_t clientId = request.client_id();
    mem[clientId] = std::make_shared<CpuMemoryHandle>(totalLen);
    CHECK_EQ(totalLen % sizeof(block.data_size()), 0U);
    msgReader->readNextBlock(mem[clientId].get()->getBuf());
  }
  msgReader.reset();
  std::vector<iovec> outputIovs;
  callback(response, outputIovs);
}

template <class T>
void DataServer::getData(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback,
                         std::vector<CpuMemHandlePtr>& mem) {
  SendDataResponse response;
  int64_t serverId = request.server_id();
  if (serverId_ < 0) {
    serverId_ = serverId;
  } else {
    CHECK_EQ(serverId_, serverId);
  }
  response.set_type(request.type());
  response.set_server_id(request.server_id());

  /* each trainer need to get all output features and label of trainers.
   * so send back all blocks in mems_. */
  std::vector<iovec> outputIovs;
  for (size_t i = 0; i < mem.size(); i++) {
    auto block = response.add_blocks();
    size_t rawMemSize = mem[i].get()->getSize();
    auto sendData = reinterpret_cast<T*>(mem[i].get()->getBuf());
    outputIovs.push_back({sendData, rawMemSize});

    block->set_total_size(rawMemSize);
    block->set_data_size(sizeof(T));
  }
  callback(response, outputIovs);
}

void DataServer::getRefGrad(const SendDataRequest& request,
                            std::unique_ptr<MsgReader>& msgReader,
                            ProtoResponseCallbackEx& callback) {
  SendDataResponse response;
  int64_t serverId = request.server_id();
  if (serverId_ < 0) {
    serverId_ = serverId;
  } else {
    CHECK_EQ(serverId_, serverId);
  }
  response.set_type(request.type());
  response.set_server_id(request.server_id());

  std::vector<iovec> outputIovs;
  size_t clientId = request.client_id();
  size_t serviceNumPerTrainer = serviceNum_ / FLAGS_num_gradient_servers;
  if (serviceNumPerTrainer == 0) {
    // if serviceNum_ < num_gradient_server, only 1 dserver is suported
    // right now.
    CHECK_EQ(serviceNum_, 1LU);
  }
  /* for each trainer  batchSize = 100, trainer num is 3, total sample
   * num is 300, if service_num = 6, then each service get 50 sample ref grad;
   *  service_num_per_trainer = service_num/trainer_num.
   *  for trainerId = i, service id = [i * service_num_per_trainer,
   *  (i + 1) * service_num_per_trainer)
   */
  CpuMemHandlePtr refGradSum;
  if (serviceNumPerTrainer == 0 ||
      clientId == request.server_id() / serviceNumPerTrainer) {
    CHECK(DATA_REFGRAD == request.type());
    size_t rawMemSize = refGradMems_[0].get()->getSize();
    CHECK_EQ(rawMemSize % sizeof(real), 0U);
    refGradSum = std::make_shared<CpuMemoryHandle>(rawMemSize);
    auto sendData = reinterpret_cast<real*>(refGradSum->getBuf());
    size_t refGradMemSize = rawMemSize / sizeof(real);
    if (serviceNumPerTrainer == 0) refGradMemSize /= FLAGS_num_gradient_servers;
    memset(sendData, 0, sizeof(real) * refGradMemSize);

    for (size_t i = 0; i < refGradMems_.size(); ++i) {
      CHECK_EQ(refGradMems_[i].get()->getSize(), rawMemSize);
      auto data = reinterpret_cast<real*>(refGradMems_[i].get()->getBuf());
      if (serviceNumPerTrainer == 0) data += refGradMemSize * clientId;
      for (size_t j = 0; j < refGradMemSize; ++j) {
        sendData[j] += data[j];
      }
    }
    auto block = response.add_blocks();
    outputIovs.push_back({sendData, rawMemSize});
    block->set_total_size(rawMemSize);
    block->set_data_size(sizeof(real));
  }

  callback(response, outputIovs);
}

void DataServer::sendData(const SendDataRequest& request,
                          std::unique_ptr<MsgReader> msgReader,
                          ProtoResponseCallbackEx callback) {
  switch (request.update_mode()) {
    case DATA_UPDATE_MODE_SET_REF:
      setData(request, msgReader, callback, refMems_);
      break;
    case DATA_UPDATE_MODE_SET_REF_LABEL:
      setData(request, msgReader, callback, refLabelMems_);
      break;
    case DATA_UPDATE_MODE_SET_REF_GRAD:
      setData(request, msgReader, callback, refGradMems_);
      break;
    case DATA_UPDATE_MODE_GET_REF:
      getData<real>(request, msgReader, callback, refMems_);
      break;
    case DATA_UPDATE_MODE_GET_REF_LABEL:
      getData<int>(request, msgReader, callback, refLabelMems_);
      break;
    case DATA_UPDATE_MODE_GET_REF_GRAD:
      getRefGrad(request, msgReader, callback);
      break;
    default:
      LOG(FATAL) << "not supported";
      break;
  }
}

void DataServer::synchronize(const SynchronizeRequest& request,
                             ProtoResponseCallback callback) {
  CHECK_LT(request.sync_object_id(), SyncObject_ARRAYSIZE);
  synchronizeBarriers_[request.sync_object_id()]->wait();
  callback(SynchronizeResponse());
}
}  // namespace paddle
