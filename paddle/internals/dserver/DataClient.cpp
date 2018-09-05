/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Logging.h"
#include "paddle/utils/StringUtil.h"
#include <vector>
#include <string.h>
#include "paddle/utils/Stat.h"
#include "DataClient.h"

P_DECLARE_string(pservers);
P_DECLARE_int32(num_gradient_servers);

namespace paddle {

DataClient::DataClient() : BaseClient(true) {}

DataClient::~DataClient() { destroy(); }

void DataClient::destroy() {
  if (clients_.empty()) {
    //  this means not initialized.
    return;
  }
  finishThreads();
  clients_.clear();
}

void DataClient::init() {
  destroy();
  initThreads();
}

void DataClient::initThreads() {
  std::vector<std::string> hosts;
  str::split(FLAGS_pservers, ',', &hosts);
  serviceNum_ = hosts.size() * FLAGS_ports_num;
  clients_.reserve(serviceNum_);
  for (size_t i = 0; i < hosts.size(); ++i) {
    for (int j = 0; j < FLAGS_ports_num; ++j) {
      LOG(INFO) << "dserver " << i * FLAGS_ports_num + j << " " << hosts[i]
                << ":" << FLAGS_data_server_port + j;
      if (FLAGS_rdma_tcp == "rdma") {
         clients_.emplace_back(hosts[i], FLAGS_data_server_port + j, F_RDMA);
      } else {
         clients_.emplace_back(hosts[i], FLAGS_data_server_port + j, F_TCP);
      }
    }
  }
  sleep(2);
  threadNum_ = serviceNum_;
  startThreads();

  gradMems_.resize(serviceNum_);
  refMems_.resize(serviceNum_ * FLAGS_num_gradient_servers);
}

void DataClient::send(int threadId) {
  LOG(INFO) << "send thread " << threadId << " started";
  int index = threadId;
  int numMyClients = divup(serviceNum_ - index, threadNum_);
  while (true) {
    SendJobPtr recvJob = sendJobQueue_[index]->dequeue();
    if (stopping_) {
      recvJobQueue_[index]->enqueue(recvJob);
      break;
    }
    for (int j = 0; j < numMyClients; ++j) {
      REGISTER_TIMER("data_client_send");
      int i = threadNum_ * j + index;
      i = calcClientId(i, serviceNum_);
      clients_[i].send("sendData", recvJob->parallelDataRequests[i],
                       recvJob->parallelInputIovs[i]);
      recvJobQueue_[index]->enqueue(recvJob);
    }
  }
}

void DataClient::recv(int threadId) {
  LOG(INFO) << "recv thread " << threadId << " started";
  int index = threadId;
  int numMyClients = divup(serviceNum_ - index, threadNum_);
  while (true) {
    std::vector<void*> bufs;
    SendDataResponse dataResponse;
    SendJobPtr recvJob = recvJobQueue_[index]->dequeue();
    if (stopping_) break;
    for (int j = 0; j < numMyClients; ++j) {
      REGISTER_TIMER("data_client_recv");
      int i = threadNum_ * j + index;
      i = calcClientId(i, serviceNum_);
      DataUpdateMode updateMode =
          recvJob->parallelDataRequests[0].update_mode();
      if (updateMode == DATA_UPDATE_MODE_GET_REF ||
          updateMode == DATA_UPDATE_MODE_GET_REF_LABEL) {
        auto msgReader = clients_[i].recv(&dataResponse);
        CHECK_EQ(msgReader->getNumBlocks(), (size_t)dataResponse.blocks_size());
        size_t totalLen = msgReader->getTotalLength();
        if (0 == totalLen) {
          continue;
        }
        /* In refMems_, the positions of blocks from one trainer are continues.
         * for example:4 servers,  2 trainers, (trainerId, serverId)
         *     [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)];
         * trainerId is also blockId, so (trainerId, serverId) position in
         * refMems_ is trainerId * serviceNum_ + serverId*/
        for (size_t blockId = 0; blockId < size_t(dataResponse.blocks_size());
             blockId++) {
          auto& recvMem =
              refMems_[blockId * serviceNum_ + dataResponse.server_id()];
          size_t length = msgReader->getNextBlockLength();
          recvMem = std::make_shared<CpuMemoryHandle>(length);
          msgReader->readNextBlock(recvMem.get()->getBuf());
        }
      } else {
        /* Each server will send back one block ref grad.*/
        auto msgReader = clients_[i].recv(&dataResponse);
        CHECK_EQ(msgReader->getNumBlocks(), (size_t)dataResponse.blocks_size());
        size_t totalLen = msgReader->getTotalLength();
        if (0 == totalLen) {
          continue;
        }
        auto& recvMem = gradMems_[dataResponse.server_id()];
        CHECK_EQ(dataResponse.blocks_size(), 1)
            << "Only one block currently support now!";
        recvMem = std::make_shared<CpuMemoryHandle>(totalLen);
        msgReader->readNextBlock(recvMem.get()->getBuf());
      }
    }
    recvSyncBarrier_->wait();
  }
}

}  // namespace paddle
