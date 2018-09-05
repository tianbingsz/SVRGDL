/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "AsgdSparseGradientMachine.h"
#include "paddle/internals/parameter/AsgdThreadUpdater.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/gradientmachines/GradientMachineMode.h"

namespace paddle {

AsgdSparseGradientMachine::AsgdSparseGradientMachine(const ModelConfig& config)
    : gradReadyBarrier_(FLAGS_trainer_count + 1),
      valueReadyBarrier_(FLAGS_trainer_count + 1) {
  ParamInitCallback mainParamInitCb = [this](int paramId, Parameter* para) {
    (void)paramId;
    para->enableType(PARAMETER_VALUE);
    para->enableType(PARAMETER_MOMENTUM);
  };

  NeuralNetwork* nn = NeuralNetwork::create(config);
  nn->init(config, mainParamInitCb);
  gradientMachine_.reset(nn);
  parameters_ = gradientMachine_->getParameters();
  CHECK(parameters_.size() == gradientMachine_->getNonStaticParameters().size())
      << "AsgdSparseGradientMachine does not support static parameters";

  for (int i = 0; i < FLAGS_trainer_count; ++i) {
    threads_.emplace_back(new AsgdSparseCpuThread(config, i, this));
  }
}

void AsgdSparseGradientMachine::start(const TrainerConfig& config,
                                      DataProviderPtr dataProvider) {
  config_ = config;
  dataProvider_ = dataProvider;

  for (auto& thread : threads_) {
    thread->start();
  }
}

void AsgdSparseGradientMachine::finish() {
  for (auto& thread : threads_) {
    thread->setStopping();
  }

  // notify worker thread to stop
  valueReadyBarrier_.wait();

  for (auto& thread : threads_) {
    thread->stop();
  }
}

void AsgdSparseGradientMachine::onPassEnd() {
  for (auto& thread : threads_) {
    thread->onPassEnd();
  }
}

void AsgdSparseGradientMachine::getStats(real& cost, int64_t& numProcessed) {
  for (auto& thread : threads_) {
    cost += thread->getCost();
    numProcessed += thread->getNumProcessed();
  }
}

void AsgdSparseGradientMachine::forward(const std::vector<Argument>& inArgs,
                                        std::vector<Argument>* outArgs,
                                        PassType passType) {
  (void)inArgs;
  (void)outArgs;
  (void)passType;
  LOG(FATAL) << "Not supported";
}

void AsgdSparseGradientMachine::backward(const UpdateCallback& callback) {
  (void)callback;
  LOG(FATAL) << "Not supported";
}

void AsgdSparseGradientMachine::forwardBackward(
    const std::vector<Argument>& inArgs, std::vector<Argument>* outArgs,
    PassType passType, const UpdateCallback& callback) {
  CHECK(inArgs.empty());
  CHECK(!outArgs);
  CHECK(passType == PASS_TRAIN);
  CHECK(!callback);

  // notify worker thread: value ready
  valueReadyBarrier_.wait();

  // wait all worker thread grad ready
  gradReadyBarrier_.wait();
}

AsgdSparseCpuThread::AsgdSparseCpuThread(
    const ModelConfig& config, int threadId,
    AsgdSparseGradientMachine* multiMachine) {
  multiMachine_ = multiMachine;
  threadId_ = threadId;
  stopping_ = false;

  auto& mainParas = multiMachine->getParameters();
  ParamInitCallback slaveParamInitCb = [&](int paramId, Parameter* para) {
    if (para->isGradSparseUpdate()) {
      para->enableType(PARAMETER_VALUE, Parameter::MAT_CACHE_ROW);
      dynamic_cast<CacheRowCpuMatrix*>(para->getMat(PARAMETER_VALUE).get())
          ->setSourceData(std::dynamic_pointer_cast<CpuVector>(
              mainParas[paramId]->getBuf(PARAMETER_VALUE)));
      para->enableType(PARAMETER_GRADIENT, Parameter::MAT_SPARSE_ROW);
      size_t height = para->getConfig().dims(0);
      para->enableIntType(PARAMETER_UPDATE_TIME, height);
    } else {
      para->enableType(PARAMETER_VALUE);
      para->enableType(PARAMETER_GRADIENT);
    }
  };

  NeuralNetwork* nn = NeuralNetwork::create(config);
  nn->init(config, slaveParamInitCb);
  gradientMachine_.reset(nn);
  parameters_ = gradientMachine_->getParameters();
}

AsgdSparseCpuThread::~AsgdSparseCpuThread() { stop(); }

void AsgdSparseCpuThread::start() {
  // opt_config works here
  parameterUpdater_.reset(new AsgdThreadUpdater(
      multiMachine_->getOptConfig(), multiMachine_->getParameters()));
  parameterUpdater_->init(parameters_);

  computeThread_.reset(new std::thread([this]() { computeThread(); }));
}

void AsgdSparseCpuThread::stop() {
  if (computeThread_) {
    computeThread_->join();
    computeThread_.reset(nullptr);
  }
}

void AsgdSparseCpuThread::computeThread() {
  LOG(INFO) << "gradComputeThread " << threadId_;

  while (true) {
    multiMachine_->waitValueReady();

    if (stopping_) break;

    parameterUpdater_->startPass();

    totalCost_ = 0;
    numProcessed_ = 0;
    int batchId = 0;
    while (trainOneBatch(batchId++)) {
    }

    parameterUpdater_->finishPass(totalCost_);

    multiMachine_->notifyGradReady();
  }
}

int64_t AsgdSparseCpuThread::trainOneBatch(int batchId) {
  DataBatch dataBatch;
  int32_t batchSize = multiMachine_->getBatchSize();
  multiMachine_->getDataProvider()->getNextBatch(batchSize, &dataBatch);
  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return 0;
  }

  const std::vector<Argument>& inArgs = dataBatch.getStreams();

  parameterUpdater_->startBatch(actualBatchSize);

  std::vector<Argument> outArgs;
  gradientMachine_->forward(inArgs, &outArgs, PASS_TRAIN);
  real cost = Argument::sumCosts(outArgs);

  UpdateCallback updateCallback =
      [this](Parameter* para) { parameterUpdater_->update(para); };
  gradientMachine_->backward(updateCallback);

  parameterUpdater_->finishBatch(cost);

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

class AsgdSparseGradientMachineMode : public IGradientMachineMode {
  // IGradientMachineMode interface
public:
  GradientMachine* create(const ModelConfig& config) {
    return new AsgdSparseGradientMachine(config);
  }

  bool shouldBeMe(const std::string& algo, size_t trainerCount,
                  bool isLocal, bool isGpu) const {
    return algo == TrainAlgorithm::AsyncSGD && !isGpu && trainerCount > 1
        && isLocal;
  }

  bool isDataMustInCpu(size_t trainerCount) const {
    return true;  // because if is not useGpu, this mode cannot happen.
  }

  bool needTrainWholeDataInOneBatch() const {
    return true;
  }
};

InitFunction __init_asgd_sparse_gradient_machine__([]{
  constexpr int kAsgdSparseCpuTraining = 2;
  IGradientMachineMode::regGradientMachineMode(
      kAsgdSparseCpuTraining, std::unique_ptr<IGradientMachineMode>(
        new AsgdSparseGradientMachineMode()));
});
}  // namespace paddle
