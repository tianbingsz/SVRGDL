/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "AsgdThreadUpdater.h"

namespace paddle {
void AsgdThreadUpdater::startPass() {
  timer_ = 0;

  // clear t0
  for (auto& para : parameters_) {
    IVectorPtr t0 = para->getIntBuf(PARAMETER_UPDATE_TIME);
    if (t0) {
      t0->zeroMem();
    }
  }
}

bool AsgdThreadUpdater::finishPass(real cost) {
  // all para W(t0) -> W(t+1)
  this->updateGradsFini(learningRate_, timer_);
  ++pass_;
  return true;
}

PassType AsgdThreadUpdater::startBatch(int64_t batchSize) {
  numSamplesProcessed_ += batchSize;
  learningRate_ = calcLearningRate(numSamplesProcessed_, pass_);

  this->copyValues();

  // grad zeroMem
  for (auto& para : parameters_) {
    // zeroMem will also clear rows for SparseRowCpuMatrix
    para->getMat(PARAMETER_GRADIENT)->zeroMem();
  }

  return PASS_TRAIN;
}

void AsgdThreadUpdater::finishBatch(real cost) { timer_++; }

void AsgdThreadUpdater::updateImpl(Parameter* para) {
  size_t pid = para->getID();

  if (para->isGradSparseUpdate()) {
    mainParameters_[pid]->updateWithGradient(
        learningRate_, para->getMat(PARAMETER_GRADIENT),
        para->getIntBuf(PARAMETER_UPDATE_TIME), timer_);
  } else {
    mainParameters_[pid]->updateWithGradient(learningRate_,
                                             para->getBuf(PARAMETER_GRADIENT));
  }
}

void AsgdThreadUpdater::copyValues() {
  for (size_t pid = 0; pid < mainParameters_.size(); ++pid) {
    const VectorPtr& value = mainParameters_[pid]->getBuf(PARAMETER_VALUE);
    if (!parameters_[pid]->isGradSparseUpdate()) {
      parameters_[pid]->getBuf(PARAMETER_VALUE)->copyFrom(*value);
    }
  }
}

void AsgdThreadUpdater::updateGradsFini(real learningRate, int currentTime) {
  for (size_t pid = 0; pid < mainParameters_.size(); ++pid) {
    if (parameters_[pid]->isGradSparseUpdate()) {
      mainParameters_[pid]->updateWithGradient(
          learningRate, parameters_[pid]->getMat(PARAMETER_GRADIENT),
          parameters_[pid]->getIntBuf(PARAMETER_UPDATE_TIME), currentTime,
          true /*fini*/);
    }
  }
}
}  // namespace paddle
