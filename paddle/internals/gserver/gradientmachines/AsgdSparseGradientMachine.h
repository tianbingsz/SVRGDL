/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/utils/Locks.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

namespace paddle {

class AsgdSparseCpuThread;

class AsgdSparseGradientMachine : public GradientMachine {
public:
  explicit AsgdSparseGradientMachine(const ModelConfig& config);
  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs, PassType passType);

  virtual void backward(const UpdateCallback& callback = nullptr);

  void forwardBackward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs, PassType passType,
                       const UpdateCallback& callback);

  virtual void onPassEnd();

  virtual void start(const TrainerConfig& config, DataProviderPtr dataProvider);

  virtual void finish();

  virtual Evaluator* makeEvaluator() { return NULL; }
  virtual void eval(Evaluator* evaluator) { (void)evaluator; }

  virtual void getStats(real& cost, int64_t& numProcessed);

  void waitValueReady() { valueReadyBarrier_.wait(); }
  void notifyGradReady() { gradReadyBarrier_.wait(); }
  DataProviderPtr getDataProvider() { return dataProvider_; }
  int getBatchSize() { return config_.opt_config().batch_size(); }
  const OptimizationConfig& getOptConfig() { return config_.opt_config(); }

protected:
  std::vector<std::unique_ptr<AsgdSparseCpuThread>> threads_;

  ThreadBarrier gradReadyBarrier_;
  ThreadBarrier valueReadyBarrier_;
  std::mutex paramMutex_;

  DataProviderPtr dataProvider_;
  TrainerConfig config_;

  // store main parameter only
  std::unique_ptr<GradientMachine> gradientMachine_;
};

class AsgdSparseCpuThread {
public:
  AsgdSparseCpuThread(const ModelConfig& config, int threadId,
                      AsgdSparseGradientMachine* multiMachine);

  ~AsgdSparseCpuThread();

  void start();
  void stop();

  void setStopping() { stopping_ = true; }

  void onPassEnd() { gradientMachine_->onPassEnd(); }

  const std::vector<ParameterPtr>& getParameters() { return parameters_; }

  real getCost() { return totalCost_; }
  int64_t getNumProcessed() { return numProcessed_; }

protected:
  void computeThread();
  int64_t trainOneBatch(int batchId);

protected:
  AsgdSparseGradientMachine* multiMachine_;
  int threadId_;  // from 0 to #threads-1
  std::unique_ptr<std::thread> computeThread_;
  bool stopping_;  // whether the thread should stop

  real totalCost_;
  int64_t numProcessed_;  // per pass

  std::unique_ptr<GradientMachine> gradientMachine_;
  std::vector<ParameterPtr> parameters_;  // = gradientMachine_->getParameters()
  std::unique_ptr<ParameterUpdater> parameterUpdater_;
};

}  // namespace paddle
