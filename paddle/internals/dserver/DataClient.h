/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include "paddle/pserver/ProtoServer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Queue.h"
#include "paddle/utils/TypeDefs.h"
#include "ParameterService.pb.h"
#include "paddle/pserver/BaseClient.h"

namespace paddle {

class DataClient : public BaseClient {
public:
  DataClient();

  virtual ~DataClient();

  void initThreads();

  void init();

  /*get data from dserver*/
  template <class DataType>
  void getData(int clientId, SendDataType type, DataType* datas, size_t size,
               DataUpdateMode mode) {
    sendData(clientId, type, mode, reinterpret_cast<DataType*>(NULL), 0);
    recvData();
    if (type == DATA_REF || type == DATA_REFLABEL) {
      cpyData(datas, refMems_, size);
    } else {
      cpyData(datas, gradMems_, size);
    }
  }

private:
  void destroy();

  /* use multi thread to send data, each thread is responsible
   * for part of clients*/
  void send(int threadId);

  /* use multi thread to receive data, each thread is responsible
   * for part of clients*/
  void recv(int threadId);

  /*copy src data to datas*/
  template <class DataType>
  void cpyData(DataType* datas, std::vector<CpuMemHandlePtr>& src,
               size_t size) {
    size_t dataOffset = 0;
    for (auto& mem : src) {
      /*copy ref or ref label*/
      if (mem) {
        CHECK_LE(dataOffset, size);
        size_t memSize =
            std::min(mem->getSize(), sizeof(DataType) * (size - dataOffset));
        CHECK_EQ(memSize % sizeof(DataType), size_t(0));
        memcpy(datas + dataOffset, mem->getBuf(), memSize);
        dataOffset += memSize / sizeof(DataType);
      }
    }
    CHECK_EQ(dataOffset, size);
  }

protected:
  std::vector<CpuMemHandlePtr> gradMems_;  // for ref grad
  std::vector<CpuMemHandlePtr> refMems_;   // for ref featrue and label
};
}  // namespace paddle
