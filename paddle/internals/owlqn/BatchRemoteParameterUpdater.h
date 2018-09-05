/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once
#include "paddle/trainer/RemoteParameterUpdater.h"

#include "TrainerOWLQN.h"

namespace paddle {

class BatchRemoteParameterUpdater : public RemoteParameterUpdater {
public:
  BatchRemoteParameterUpdater(OptimizationConfig config, int expectedPassCount);

  virtual void init(std::vector<ParameterPtr>& parameters);
  virtual void startPass() {
    passCost_ = 0;
    passAccepted_ = false;
  }
  virtual bool finishPass(real cost);
  virtual void finishBatch(real cost);

protected:
  virtual void controller();
  virtual void updateImpl(Parameter* para);
  double passCost_;
  bool passAccepted_;
};

}  // namespace paddle
