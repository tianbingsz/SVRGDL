/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Stat.h"
#include "BatchGradientMachine.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/gradientmachines/GradientMachineMode.h"
#include "paddle/utils/Util.h"

namespace paddle {

BatchGradientMachine::BatchGradientMachine(const ModelConfig& config,
                                           bool useGpu)
    : useGpu_(useGpu),
      gradReadyBarrier_(FLAGS_trainer_count + 1),
      valueReadyBarrier_(FLAGS_trainer_count + 1) {
  if (useGpu_) {
    numDevices_ = hl_get_device_count();
    CHECK(FLAGS_trainer_count <= numDevices_)
        << "trainer_count is bigger than the number of device";
  }

  for (int i = 0; i < FLAGS_trainer_count; ++i) {
    threads_.emplace_back(new BatchCpuThread(config, i, this, useGpu_,
                                             useGpu_ ? i % numDevices_ : 0));

    if (i == 0) {
      if (!useGpu_) {
        parameters_ = threads_[0]->getParameters();
      } else {
        parameters_.reserve(threads_[0]->getParameters().size());
        for (size_t pid = 0; pid < threads_[0]->getParameters().size(); pid++) {
          auto parameter = std::make_shared<Parameter>(
              threads_[0]->getParameters()[pid]->getConfig(),
              /* useGpu= */ false,
              /* doInit= */ false);
          parameter->enableType(PARAMETER_VALUE);
          if (!parameter->isStatic()) {
            parameter->enableType(PARAMETER_GRADIENT);
            parameter->enableType(PARAMETER_GRADIENT_SUM);
          }
          parameter->setID(pid);
          parameters_.push_back(parameter);
        }
      }
    }
  }
}

void BatchGradientMachine::start(const TrainerConfig& config,
                                 DataProviderPtr dataProvider) {
  config_ = config;
  dataProvider_ = dataProvider;

  for (auto& thread : threads_) {
    thread->start();
  }
}

void BatchGradientMachine::finish() {
  for (auto& thread : threads_) {
    thread->setStopping();
  }

  // notify worker thread to stop
  valueReadyBarrier_.wait();

  for (auto& thread : threads_) {
    thread->stop();
  }
}

void BatchGradientMachine::onPassEnd() {
  for (auto& thread : threads_) {
    thread->onPassEnd();
  }
}

void BatchGradientMachine::sumGrad(int paramId, VectorPtr gradBuf) {
  std::lock_guard<std::mutex> lock(paramMutex_);
  if (!useGpu_) {
    parameters_[paramId]->getBuf(PARAMETER_GRADIENT_SUM)->add(*gradBuf);
  } else {
    parameters_[paramId]->getBuf(PARAMETER_GRADIENT)->copyFrom(*gradBuf);
    parameters_[paramId]
        ->getBuf(PARAMETER_GRADIENT_SUM)
        ->add(*parameters_[paramId]->getBuf(PARAMETER_GRADIENT));
  }
}

void BatchGradientMachine::getStats(real& cost, int64_t& numProcessed) {
  for (auto& thread : threads_) {
    cost += thread->getCost();
    numProcessed += thread->getNumProcessed();
  }
}

void BatchGradientMachine::forward(const std::vector<Argument>& inArgs,
                                   std::vector<Argument>* outArgs,
                                   PassType passType) {
  (void)inArgs;
  (void)outArgs;
  (void)passType;
  LOG(FATAL) << "Not supported";
}

void BatchGradientMachine::backward(const UpdateCallback& callback) {
  (void)callback;
  LOG(FATAL) << "Not supported";
}

void BatchGradientMachine::forwardBackward(const std::vector<Argument>& inArgs,
                                           std::vector<Argument>* outArgs,
                                           PassType passType,
                                           const UpdateCallback& callback) {
  CHECK(inArgs.empty());
  CHECK(!outArgs);
  CHECK(passType == PASS_TRAIN);
  CHECK(!callback);

  // notify worker thread: value ready
  valueReadyBarrier_.wait();

  // wait all worker thread grad ready
  gradReadyBarrier_.wait();
}

BatchCpuThread::BatchCpuThread(const ModelConfig& config, int threadId,
                               BatchGradientMachine* multiMachine, bool useGpu,
                               int deviceId) {
  multiMachine_ = multiMachine;
  threadId_ = threadId;
  stopping_ = false;
  useGpu_ = useGpu;
  deviceId_ = deviceId;

  ParamInitCallback mainParamInitCb = [this](int paramId, Parameter* para) {
    (void)paramId;
    para->enableType(PARAMETER_VALUE);
    if (para->isGradShared()) {
      para->enableType(PARAMETER_GRADIENT);
      para->enableSharedType(PARAMETER_GRADIENT_SUM,
                             para->getBuf(PARAMETER_GRADIENT),
                             Parameter::MAT_NORMAL_SHARED);
    } else if (!para->isStatic()) {
      para->enableType(PARAMETER_GRADIENT);
      para->enableType(PARAMETER_GRADIENT_SUM);
    }
  };

  ParamInitCallback slaveParamInitCb = [this](int paramId, Parameter* para) {
    para->enableSharedType(
        PARAMETER_VALUE,
        multiMachine_->getParameters()[paramId]->getBuf(PARAMETER_VALUE));
    if (para->isGradShared()) {
      para->enableSharedType(
          PARAMETER_GRADIENT,
          multiMachine_->getParameters()[paramId]->getBuf(PARAMETER_GRADIENT),
          multiMachine_->getParameters()[paramId]->getMat(PARAMETER_GRADIENT));
    } else if (!para->isStatic()) {
      para->enableType(PARAMETER_GRADIENT);
    }
  };

  ParamInitCallback gpuParamInitCb = [this](int paramId, Parameter* para) {
    (void)paramId;
    para->enableType(PARAMETER_VALUE);
    if (!para->isStatic()) {
      para->enableType(PARAMETER_GRADIENT);
    }
  };

  int devId = 0;
  if (useGpu_) {
    devId = hl_get_device();
    hl_set_device(deviceId_);
  }

  NeuralNetwork* nn = NeuralNetwork::create(config);
  if (useGpu_) {
    nn->init(config, gpuParamInitCb);
  } else {
    nn->init(config, threadId_ == 0 ? mainParamInitCb : slaveParamInitCb);
  }
  gradientMachine_.reset(nn);
  parameters_ = gradientMachine_->getParameters();

  if (useGpu_) {
    hl_set_device(devId);
  }
}

BatchCpuThread::~BatchCpuThread() { stop(); }

void BatchCpuThread::start() {
  computeThread_.reset(new std::thread([this]() { computeThread(); }));
}

void BatchCpuThread::stop() {
  if (computeThread_) {
    computeThread_->join();
    computeThread_.reset(nullptr);
  }
}

void BatchCpuThread::computeThread() {
  LOG(INFO) << "gradComputeThread " << threadId_;

  if (useGpu_) {
    hl_init(deviceId_);
  }

  while (true) {
    multiMachine_->waitValueReady();

    if (stopping_) break;

    /* copy value from batch matchine*/
    if (useGpu_) {
      for (size_t pid = 0; pid < parameters_.size(); ++pid) {
        if (parameters_[pid]->isValueUpdated()) continue;
        if (parameters_[pid]->isStatic()) {
          /* copy static parameters only once */
          parameters_[pid]->setValueUpdated();
        }
        parameters_[pid]
            ->getBuf(PARAMETER_VALUE)
            ->copyFrom(
                *multiMachine_->getParameters()[pid]->getBuf(PARAMETER_VALUE));
      }
    }

    totalCost_ = 0;
    numProcessed_ = 0;
    int batchId = 0;
    while (trainOneBatch(batchId++)) {
      if (useGpu_) continue;
      for (size_t pid = 0; pid < parameters_.size(); ++pid) {
        if (parameters_[pid]->isGradShared() || parameters_[pid]->isStatic()) {
          continue;
        }
        multiMachine_->sumGrad(pid,
                               parameters_[pid]->getBuf(PARAMETER_GRADIENT));
        parameters_[pid]->getBuf(PARAMETER_GRADIENT)->zeroMem();
      }
    }

    if (useGpu_) {
      for (size_t pid = 0; pid < parameters_.size(); ++pid) {
        if (parameters_[pid]->isStatic()) continue;
        multiMachine_->sumGrad(pid,
                               parameters_[pid]->getBuf(PARAMETER_GRADIENT));
        parameters_[pid]->getBuf(PARAMETER_GRADIENT)->zeroMem();
      }
    }

    multiMachine_->notifyGradReady();
  }
}

int64_t BatchCpuThread::trainOneBatch(int batchId) {
  DataBatch dataBatch;
  int32_t batchSize = multiMachine_->getBatchSize();
  multiMachine_->getDataProvider()->getNextBatch(batchSize, &dataBatch);
  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return 0;
  }

  std::vector<Argument> outArgs;
  const std::vector<Argument>& inArgs = dataBatch.getStreams();

  gradientMachine_->forward(inArgs, &outArgs, PASS_TRAIN);
  gradientMachine_->backward(nullptr);

  real cost = Argument::sumCosts(outArgs);
  totalCost_ += cost;
  numProcessed_ += actualBatchSize;

  if ((batchId + 1) % FLAGS_log_period == 0) {
    real avgBatchCost = cost / actualBatchSize;
    real avgTotalCost = totalCost_ / numProcessed_;
    LOG(INFO) << " Batch=" << batchId + 1 << " Thread=" << threadId_
              << " samples=" << numProcessed_
              << " AvgBatchCost=" << avgBatchCost
              << " AvgTotalCost=" << avgTotalCost;
  }

  return actualBatchSize;
}

class BatchGradientMachineMode : public IGradientMachineMode {
  // IGradientMachineMode interface
public:
  GradientMachine* create(const ModelConfig& config) {
    return new BatchGradientMachine(config);
  }

  bool shouldBeMe(const std::string& algo, size_t trainerCount,
                  bool isLocal, bool isGpu) const {
    return algo == TrainAlgorithm::OWLQN && trainerCount > 1;
  }

  bool isDataMustInCpu(size_t trainerCount) const {
    return trainerCount > 1;
  }

  bool needTrainWholeDataInOneBatch() const {
    return true;
  }
};

InitFunction __init_batch_gradient_machine__([]{
  constexpr int kBatchMultiThreadTraining = 1;
  IGradientMachineMode::regGradientMachineMode(
      kBatchMultiThreadTraining, std::unique_ptr<IGradientMachineMode>(
        new BatchGradientMachineMode()));
});

}  // namespace paddle
