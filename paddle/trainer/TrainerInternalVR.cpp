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


#include "TrainerInternalVR.h"

#include <fenv.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

#include <google/protobuf/text_format.h>

#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"
#include "paddle/utils/GlobalConstants.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
#include "paddle/gserver/layers/ValidationLayer.h"

#include "ThreadParameterUpdater.h"
#include "RemoteParameterUpdaterVR.h"

namespace paddle {

void TrainerInternalVR::init(const std::shared_ptr<TrainerConfigHelper> &config,
                           const GradientMachinePtr &gradientMachine,
                           std::unique_ptr<TrainerInternalConfig> &&intconfig,
                           const std::shared_ptr<TrainerStats> &stats,
                           bool testing) {
    config_ = config;
    intconfig_ = std::move(intconfig);
    stats_ = stats;
    createParameterUpdater(testing);

    gradientMachine_ = gradientMachine;
    if (!gradientMachine) {
      gradientMachine_.reset(GradientMachine::create(
        config_->getConfig().model_config(), intconfig_->mode,
        parameterUpdater_->getParameterTypes()));
    }
}

void TrainerInternalVR::calcGradOneBatch(int64_t batchId,
                                         const DataBatch& dataBatch) {
  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return;
  }

  std::vector<Argument> outArgs;
  const std::vector<Argument>& inArgs = dataBatch.getStreams();

  // todo, delete
  std::vector<ParaStat> paraStats;
  paraStats.resize(gradientMachine_->getParameters().size());
  UpdateCallback updateCallback =
      [this, &paraStats](Parameter* para) {
    auto& grad = para->getBuf(PARAMETER_GRADIENT);
    paraStats[para->getID()].avgAbsGrad = grad->getAbsSum() / para->getSize();
    paraStats[para->getID()].maxAbsGrad = grad->getAbsMax();
    if (intconfig_->local) {
      // accumulate gradients
      para->getBuf(PARAMETER_GRADIENT_SUM)->add(
                *para->getBuf(PARAMETER_GRADIENT));
      para->clearGradient();
    }
  };
  // todo, support prefetch for sparse remote update
  {
    REGISTER_TIMER("forwardBackwardGradSum");
    gradientMachine_->forwardBackward(
            inArgs, &outArgs, PASS_TRAIN, updateCallback);
  }

  real cost = Argument::sumCosts(outArgs);
  *stats_ += { actualBatchSize, cost };

  if ((batchId + 1) % intconfig_->log_period == 0) {
    LOG(INFO) << " Batch=" << batchId + 1 << " "
              << *stats_;
  }
}

void TrainerInternalVR::trainOneBatch(int64_t batchId,
                                      const DataBatch& dataBatch) {
  int64_t actualBatchSize = dataBatch.getSize();
  if (actualBatchSize == 0) {
    return;
  }

  // todo, delete
  std::vector<ParaStat> paraStats;
  paraStats.resize(gradientMachine_->getParameters().size());

  const std::vector<Argument>& inArgs = dataBatch.getStreams();
  std::vector<Argument> outArgs;

  PassType passType = parameterUpdater_->startBatch(actualBatchSize);

  UpdateCallback updateCallback =
      [this, &paraStats](Parameter* para) {
    auto& grad = para->getBuf(PARAMETER_GRADIENT);
    paraStats[para->getID()].avgAbsGrad = grad->getAbsSum() / para->getSize();
    paraStats[para->getID()].maxAbsGrad = grad->getAbsMax();
    parameterUpdater_->update(para);
  };

  {
#ifndef PADDLE_DISABLE_TIMER
    Timer timer;
    timer.start();
#endif
    REGISTER_TIMER("forwardBackward for Variance Reduction");
    // w <- snapshot w_s , P_SNAPSHOT_VALUE -> P_VALUE
    swapParameter();  // SNAPSHOT <-> VALUE
    // \partial f_b(w_s)
    gradientMachine_->forwardBackward(
            inArgs, &outArgs, passType, nullptr);
    // g = - \partial f_b(w_s)
    negGradients();

    // w <- w_t, P_VALUE -> P_SNAPSHOT_VALUE
    swapParameter();  // SNAPSHOT <-> VALUE
    // g = \prtial f_b(w_t) - \partial f_b(w_s)
    gradientMachine_->forwardBackward(
            inArgs, &outArgs, passType, updateCallback);

#ifndef PADDLE_DISABLE_TIMER
    timer.stop();
    parameterUpdater_->setForwardbackwardTime(timer.get());
#endif
  }

  real cost = 0;
  {
    REGISTER_TIMER("sumCost");
    cost = Argument::sumCosts(outArgs);
  }

  if (batchId % intconfig_->log_period == 0) {
    currentEvaluator_->start();
    stats_->resetCurrentStat();
  }
  {
    REGISTER_TIMER("eval");
    gradientMachine_->eval(currentEvaluator_);
    gradientMachine_->eval(evaluator_);
  }

  *stats_ += { actualBatchSize, cost };
  {
    REGISTER_TIMER("finishBatch");
    parameterUpdater_->finishBatch(cost);
  }

  if ((batchId + 1) % intconfig_->log_period == 0) {
    currentEvaluator_->finish();

    if (intconfig_->dot_period > 0) {
      std::cerr << std::endl;
    }
    LOG(INFO) << " Batch=" << batchId + 1 << " "
              << *stats_
              << " Eval: " << *evaluator_
              << " CurrentEval: " << *currentEvaluator_;
  } else if (intconfig_->dot_period > 0 &&
            (batchId + 1) % intconfig_->dot_period == 0) {
    std::cerr << ".";
  }
}

// P_VALUE <---> P_SNAPSHOT_VALUE
void TrainerInternalVR::swapParameter() {
  auto& parameters = gradientMachine_->getParameters();
  for (auto& para : parameters) {
    para->getBuf(PARAMETER_SNAPSHOT_VALUE)->deepSwap(
            *para->getBuf(PARAMETER_VALUE));
  }
}

// copy fromType parameter (val) to toType
void TrainerInternalVR::copyParameter(ParameterType fromType,
                                      ParameterType toType) {
  if (fromType == toType) {
    return;
  }
  auto& parameters = gradientMachine_->getParameters();
  for (auto& para : parameters) {
    para->getBuf(toType)->copyFrom(*para->getBuf(fromType));
  }
}

// grad = - grad
void TrainerInternalVR::negGradients() {
  auto& parameters = gradientMachine_->getParameters();
  for (auto& para : parameters) {
    para->getBuf(PARAMETER_GRADIENT)->neg();
  }
}

// grad_sum = 0 or grad = 0
void TrainerInternalVR::clearGradients(ParameterType parameterType) {
  auto& parameters = gradientMachine_->getParameters();
  for (auto& para : parameters) {
    para->getBuf(parameterType)->zeroMem();
  }
}

void TrainerInternalVR::showParameterStats(const std::vector<ParaStat>&
                                        paraStats) {
  std::vector<ParameterPtr>& parameters = gradientMachine_->getParameters();
  for (auto& parameter : parameters) {
    SetDevice device(parameter->getDeviceId());
    real sum = parameter->getBuf(PARAMETER_VALUE)->getAbsSum();
    real snapSum = parameter->getBuf(PARAMETER_SNAPSHOT_VALUE)->getAbsSum();
    const auto& lr = parameter->getBuf(PARAMETER_LEARNING_RATE);
    std::ostringstream osLrHistogram;
    if (lr) {
      if (VLOG_IS_ON(2)) {
        osLrHistogram << " lr_histogram: ";
        lr->histogram(osLrHistogram);
      } else {
        osLrHistogram << " max_lr=" << std::setw(11) << lr->getMax()
                      << " min_lr=" << std::setw(11) << lr->getMin()
                      << " avg_lr=" << std::setw(11)
                      << lr->getSum() / parameter->getSize();
      }
    }
    int pid = parameter->getID();
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
              << std::setw(20) << parameter->getName()
              << " avg_val=" << std::setw(11) << sum / parameter->getSize()
              << " avg_sval=" << std::setw(11)
              << snapSum / parameter->getSize()
              << " avg_grad=" << std::setw(11) << paraStats[pid].avgAbsGrad
              << " max_grad=" << std::setw(11) << paraStats[pid].maxAbsGrad
              << osLrHistogram.str();
  }
}

void TrainerInternalVR::createParameterUpdater(bool testing) {
  const std::string& alg = config_->getOptConfig().algorithm();
  if (intconfig_->local) {
    LOG(INFO) << "Creating Local Parameter Updater for SVRG";
    CHECK_EQ(config_->getOptConfig().num_batches_per_send_parameter(), 1)
        << "num_batches_per_send_parameter should be one in local mode!";
    parameterUpdater_.reset(new SgdLocalUpdater(config_->getOptConfig()));
  } else if (alg == TrainAlgorithm::SVRG) {
    LOG(INFO) << "Creating Remote Parameter Updater for SVRG";
    parameterUpdater_.reset(new VRRemoteParameterUpdater(
                config_->getOptConfig(),
                intconfig_->num_passes));
  } else {
    LOG(FATAL) << "Unsupported algorithm in local mode: " << alg;
  }
}

}  // namespace paddle
