/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "RemoteParameterUpdaterVR.h"
#include "Trainer.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/GlobalConstants.h"

namespace paddle {

VRRemoteParameterUpdater::VRRemoteParameterUpdater(
    OptimizationConfig config, int passCount)
    : RemoteParameterUpdater(config, passCount) {
  addParameterType(PARAMETER_GRADIENT_SUM);
  addParameterType(PARAMETER_SNAPSHOT_VALUE);
}

void VRRemoteParameterUpdater::init(std::vector<ParameterPtr>& parameters) {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::SVRG) << "SVRG not supported";
  ParameterUpdater::init(parameters);

  if (!FLAGS_use_gpu) {
    cpuParameters_ = parameters;
  } else {
    for (auto& parameter : parameters) {
      cpuParameters_.emplace_back(new Parameter(parameter->getConfig(), false));
      cpuParameters_.back()->setID(parameter->getID());
      cpuParameters_.back()->enableType(PARAMETER_GRADIENT_SUM);
      cpuParameters_.back()->enableType(PARAMETER_SNAPSHOT_VALUE);
    }
  }

  parameterClient_.reset(new ParameterClient2(separateSendAndRecv_));
  parameterClient_->init(cpuParameters_);
  parameterClient_->setTrainerId(FLAGS_trainer_id);

  if (FLAGS_trainer_id == 0) {
    parameterClient_->setConfig(config_);
    copyParametersFromDevice(PARAMETER_VALUE);
    parameterClient_->setParameter();
    parameterClient_->setStatus(PSERVER_STATUS_PARAMETER_READY);
    startController();
    useApplyInPserver_ = useApplyInPserver(config_);
  } else {
    parameterClient_->waitForStatus(PSERVER_STATUS_PARAMETER_READY);
    parameterClient_->getParameter();
    copyParametersToDevice(PARAMETER_VALUE);
  }
}

void VRRemoteParameterUpdater::controller() {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::SVRG) << "SVRG not supported";
  ParameterClient2 client(false);
  client.init(cpuParameters_);
  while (true) {
    /*start pass for full gradient*/ {
      client.waitPassStart();

      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_START_PASS);
      client.doOperation(ops,
                         /* waitForGradient= */ false,
                         /* sendBackarameter= */ false,
                         /* releasePass= */ false);
    }

    // OP_COPY_SUM_GRAD, copy aggregate grad as grad sum
    {
       PreparedOperations ops;
       PServerVector grad = { PARAMETER_GRADIENT };
       PServerVector gradSum = { PARAMETER_GRADIENT_SUM };
       ops.addOperation(PSERVER_OP_COPY_ZERO, grad, gradSum);
       client.doOperation(ops, true, true, false);
    }

    /*start pass for each mini-batch*/ {
      client.waitPassStart();

      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_START_PASS);
      client.doOperation(ops, false, false, false);
    }
    while (true) {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_SGD);
      client.doOperation(ops, true, true, false);
      if (client.isPassFinish()) {
        break;
      }
    }

    /*finish pass*/ {
      PreparedOperations ops;
      ops.addOperation(PSERVER_OP_FINISH_PASS);
      client.doOperation(ops, true, true, true);
    }

    if (++passCount_ == expectedPassCount_) {
      break;
    }
  }
}

void VRRemoteParameterUpdater::finishBatch(real cost) {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::SVRG) << "SVRG not supported";
  copyParametersFromDevice(PARAMETER_GRADIENT);

  {
    REGISTER_TIMER("sendAndRecv_dense");
    parameterClient_->sendAndReceiveParameter(PSERVER_UPDATE_MODE_ADD_GRADIENT,
                                              PARAMETER_GRADIENT, // sendType
                                              batchSize_,
                                              0,  // cost = 0
                                              true // sendBackParameter);
  }

  copyParametersToDevice(PARAMETER_VALUE);

  for (auto& para : parameters_) {
    SetDevice device(para->getDeviceId());
    para->clearGradient();
  }
}

void VRRemoteParameterUpdater::startPass() {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::SVRG) << "SVRG not supported";
  parameterClient_->waitPassStart();
}

bool VRRemoteParameterUpdater::finishPass(real cost) {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::SVRG) << "SVRG not supported";
  parameterClient_->waitPassFinish();
  parameterClient_->getParameter();
  copyParametersToDevice(PARAMETER_VALUE);

  isFirstPass_ = false;
  return true;
}

}  // namespace paddle
