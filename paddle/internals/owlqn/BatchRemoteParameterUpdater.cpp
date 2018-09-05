/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#include "BatchRemoteParameterUpdater.h"
#include "paddle/utils/Util.h"

namespace paddle {
BatchRemoteParameterUpdater::BatchRemoteParameterUpdater(
    OptimizationConfig config, int passCount)
    : RemoteParameterUpdater(config, passCount), passCost_(0) {
  addParameterType(PARAMETER_GRADIENT_SUM);
}

void BatchRemoteParameterUpdater::init(std::vector<ParameterPtr>& parameters) {
  RemoteParameterUpdater::init(parameters);
  if (FLAGS_use_gpu) {
    for (auto& parameter : cpuParameters_) {
      parameter->enableType(PARAMETER_GRADIENT_SUM);
    }
  }
}

void BatchRemoteParameterUpdater::controller() {
  CHECK_EQ(config_.algorithm(), TrainAlgorithm::OWLQN) << "not supported";

  ParameterClient2 client(false);
  client.init(cpuParameters_);

  OWLQN optimizer(client, config_, expectedPassCount_);
  optimizer.init();
  optimizer.train(passAccepted_);
  optimizer.deinit();
}

void BatchRemoteParameterUpdater::updateImpl(Parameter* para) {
  // accumulate gradients
  para->getBuf(PARAMETER_GRADIENT_SUM)->add(*para->getBuf(PARAMETER_GRADIENT));
  para->clearGradient();
}

void BatchRemoteParameterUpdater::finishBatch(real cost) { passCost_ += cost; }

bool BatchRemoteParameterUpdater::finishPass(real cost) {
  passCount_++;
  ParameterUpdateMode mode = PSERVER_UPDATE_MODE_ADD_GRADIENT;
  ParameterType sendType = PARAMETER_GRADIENT_SUM;
  copyParametersFromDevice(sendType);

  parameterClient_->sendAndReceiveParameter(mode, sendType, batchSize_,
                                            passCost_ + cost,
                                            true);  // sendBackParameter = true

  copyParametersToDevice(PARAMETER_VALUE);

  for (auto& para : parameters_) {
    para->getBuf(sendType)->zeroMem();
  }

  return passAccepted_;
}

InitFunction __init_batch_remote_parameter_updater__([]{
  ParameterUpdaterCreators::addCreator(
      [](const std::string& algo, const OptimizationConfig& optConf,
      bool isLocal, size_t numPasses) -> ParameterUpdater* {
    if (algo == TrainAlgorithm::OWLQN && !isLocal) {
      CHECK(!optConf.use_sparse_remote_updater())
          << "OWLQN can not work with sparse_remote_update setting, "
          << "disable sparse_remote_update in config file";
      return new BatchRemoteParameterUpdater(optConf, numPasses);
    } else {
      return nullptr;
    }
  });
});
}  // namespace paddle
