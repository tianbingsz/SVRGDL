/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */



#pragma once

#include <memory>
#include <vector>

#include "paddle/parameter/LearningRateScheduler.h"
#include "paddle/parameter/ParameterUpdaterBase.h"

namespace paddle {
class AsgdThreadUpdater : public ParameterUpdater {
public:
  AsgdThreadUpdater(const OptimizationConfig& optConfig,
                    std::vector<ParameterPtr>& mainParameters)
      : config_(optConfig),
        numSamplesProcessed_(0),
        learningRateScheduler_(LearningRateScheduler::create(optConfig)),
        pass_(0),
        mainParameters_(mainParameters) {}

  virtual ~AsgdThreadUpdater() {}

  real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return learningRateScheduler_->calcLearningRate(numSamplesProcessed, pass);
  }

  virtual void startPass();
  virtual bool finishPass(real cost);

  virtual PassType startBatch(int64_t batchSize);
  virtual void finishBatch(real cost);

protected:
  void copyValues();
  void updateGradsFini(real learningRate, int currentTime);
  virtual void updateImpl(Parameter* para);

  OptimizationConfig config_;
  int64_t numSamplesProcessed_;
  real learningRate_;
  int32_t timer_;
  std::unique_ptr<LearningRateScheduler> learningRateScheduler_;
  int64_t pass_;  // current training pass (starting from 0)

  std::vector<ParameterPtr>& mainParameters_;
};

}  // namespace paddle
