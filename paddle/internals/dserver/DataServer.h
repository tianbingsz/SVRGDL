/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "paddle/utils/Locks.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"
#include "paddle/utils/TypeDefs.h"
#include "paddle/math/Vector.h"
#include "paddle/pserver/ProtoServer.h"
#include "ParameterService.pb.h"

P_DECLARE_int32(port);

namespace paddle {

class DataServer : public ProtoServer {
  /* DataServer is responsible for other data synchronization and
   * callback except model parameters*/
public:
  explicit DataServer(const std::string& addr,
                      int port,
                      int rdamCpu = -1);

  ~DataServer() {}

  bool init();

  /* responsible for receive data and send back */
  void sendData(const SendDataRequest& request,
                std::unique_ptr<MsgReader> msgReader,
                ProtoResponseCallbackEx callback);

  /* responsible for multi machine synchronize */
  void synchronize(const SynchronizeRequest& request,
                   ProtoResponseCallback callback);

private:
  void setData(const SendDataRequest& request,
               std::unique_ptr<MsgReader>& msgReader,
               ProtoResponseCallbackEx& callback,
               std::vector<CpuMemHandlePtr>& mem);

  template <class T>
  void getData(const SendDataRequest& request,
               std::unique_ptr<MsgReader>& msgReader,
               ProtoResponseCallbackEx& callback,
               std::vector<CpuMemHandlePtr>& mem);

  void getRefGrad(const SendDataRequest& request,
                  std::unique_ptr<MsgReader>& msgReader,
                  ProtoResponseCallbackEx& callback);

protected:
  std::vector<CpuMemHandlePtr> refMems_;       // for ref feature
  std::vector<CpuMemHandlePtr> refLabelMems_;  // for ref label
  std::vector<CpuMemHandlePtr> refGradMems_;   // for ref grad
  std::vector<std::unique_ptr<ThreadBarrier>> synchronizeBarriers_;
  std::atomic<int> serverId_;
  size_t serviceNum_;  // The number of data services.
};
}  // namespace paddle
