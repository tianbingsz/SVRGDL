/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/utils/Locks.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

namespace paddle {

class BatchCpuThread;

class BatchGradientMachine : public GradientMachine {
public:
  explicit BatchGradientMachine(const ModelConfig& config,
                                bool useGpu = FLAGS_use_gpu);
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
  void sumGrad(int paramId, VectorPtr gradBuf);

protected:
  bool useGpu_;
  int numDevices_; /* number of gpu devices */
  std::vector<std::unique_ptr<BatchCpuThread>> threads_;

  ThreadBarrier gradReadyBarrier_;
  ThreadBarrier valueReadyBarrier_;
  std::mutex paramMutex_;

  DataProviderPtr dataProvider_;
  TrainerConfig config_;
};

class BatchCpuThread {
public:
  BatchCpuThread(const ModelConfig& config, int threadId,
                 BatchGradientMachine* multiMachine, bool useGpu = false,
                 int deviceId = 0);

  ~BatchCpuThread();

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
  BatchGradientMachine* multiMachine_;
  int threadId_;  // from 0 to #threads-1
  bool useGpu_;
  int deviceId_;  // the GPU device Id which the computeThread_ used
  std::unique_ptr<std::thread> computeThread_;
  bool stopping_;  // whether the thread should stop

  real totalCost_;
  int64_t numProcessed_;

  std::unique_ptr<GradientMachine> gradientMachine_;
  std::vector<ParameterPtr> parameters_;  // = gradientMachine_->getParameters()
};

}  // namespace paddle
