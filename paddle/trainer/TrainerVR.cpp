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
#include "paddle/gserver/gradientmachines/GradientMachineMode.h"
#include "paddle/gserver/layers/ValidationLayer.h"
#include "TesterConfig.h"
#include "TrainerVR.h"
#include "TrainerInternalVR.h"
#include "ThreadParameterUpdater.h"
#include "RemoteParameterUpdater.h"
#include "TrainerConfigHelper.h"

P_DECLARE_string(config);
P_DECLARE_int32(test_period);

P_DECLARE_bool(local);

P_DECLARE_bool(test_all_data_in_one_period);

P_DECLARE_int32(average_test_period);

P_DECLARE_int32(saving_period);
P_DECLARE_int64(saving_period_by_batches);
P_DECLARE_string(save_dir);
P_DECLARE_int32(start_pass);
P_DECLARE_int32(test_pass);
P_DECLARE_int32(test_wait);
P_DECLARE_bool(with_cost);
P_DECLARE_bool(distribute_test);

P_DECLARE_int32(num_passes);

P_DECLARE_string(config_args);

P_DECLARE_bool(save_only_one);

P_DECLARE_string(feat_file);
P_DECLARE_string(predict_output_dir);
P_DECLARE_string(model_list);

namespace paddle {

void TrainerVR::init(const std::shared_ptr<TrainerConfigHelper> &config,
                   bool testing,
                   const std::shared_ptr<GradientMachine> &gradientMachine,
                   const std::shared_ptr<DataProvider> &dataProvider,
                   const std::shared_ptr<DataProvider> &testDataProvider) {
  this->stats_ = std::make_shared<TrainerStats>();

  config_ = config;

  config_->updateConfigFromFlags();

  CHECK(TrainAlgorithm::isValid(config_->getOptConfig().algorithm()))
      << "invalid algorithm configuration: "
      << config_->getOptConfig().algorithm();

  if (config_->getOptConfig().algorithm() == TrainAlgorithm::SVRG) {
    mode_ = GradientMachine::kSVRG;
    LOG(INFO) << "trainer mode: SVRG";
  } else {
    mode_ = GradientMachine::kNormal;
    LOG(INFO) << "local : " << FLAGS_local << ",trainer mode: Normal";
  }

  // initialize trainer internal
  trainerInternal_->init(config_, gradientMachine,
                        TrainerInternalConfig::createFromMode(mode_),
                        stats_, false);
  std::unique_ptr<ParameterUtilConfig> paramConfig(
          new ParameterUtilConfig(FLAGS_save_only_one,
                                  FLAGS_saving_period,
                                  FLAGS_loadsave_parameters_in_pserver,
                                  FLAGS_config));

  paramUtil_.reset(
      new paddle::ParameterUtil(
          config_,
          std::move(paramConfig),
          trainerInternal_->getGradientMachine(),
          trainerInternal_->getParameterUpdater()));


  bool gpuData = FLAGS_use_gpu && (!FLAGS_parallel_nn) &&
                 (!IGradientMachineMode::dataMustInCpu(mode_,
                                                       FLAGS_trainer_count));

  dataProvider_ = dataProvider;
  if (!dataProvider_ && config_->hasDataConfig()) {
    dataProvider_.reset(DataProvider::create(*config_, *config_, gpuData));
  }
  if (dataProvider_) {
    evaluator_.reset(trainerInternal_->getGradientMachine()->makeEvaluator());
    currentEvaluator_.reset(
        trainerInternal_->getGradientMachine()->makeEvaluator());
  }

  testDataProvider_ = testDataProvider;
  if (!testDataProvider_ && config_->hasTestDataConfig()) {
    testDataProvider_.reset(
        DataProvider::create(config_->getTestDataConfig(), *config_, gpuData));
  }
  if (testDataProvider_) {
    tester_.reset(new Tester(config_, createTesterConfig(),
                 trainerInternal_->getGradientMachine(),
                 trainerInternal_->getParameterUpdater(),
                 testDataProvider_));
  }

  if (paramUtil_->tryLoadParametersFromConfig()) {
    // load from config already.
  } else {
    trainerInternal_->getGradientMachine()->randParameters();
  }

  // Only non static parameters need to be updated
  std::vector<ParameterPtr>& parameters =
      trainerInternal_->getGradientMachine()->getNonStaticParameters();
  if (trainerInternal_->getParameterUpdater()) {
    trainerInternal_->getParameterUpdater()->init(parameters);
  }

  // set current evaluator and evalutor
  trainerInternal_->setCurrentEvaluator(currentEvaluator_.get());
  trainerInternal_->setEvaluator(evaluator_.get());
}

void TrainerVR::calculateFullGradient(int passId) {
  this->stats_->reset();
  int64_t batchId = 0;
  int32_t batchSize = config_->getOptConfig().batch_size();

  trainerInternal_->getParameterUpdater()->startPass();
  trainerInternal_->getParameterUpdater()->startBatch(0);
  size_t passSize = 0;
  while (true) {
    DataBatch dataBatch;

    int64_t num = 0;
    {
      REGISTER_TIMER("getTrainBatchFullGrad");
      num = dataProvider_->getNextBatch(batchSize, &dataBatch);
    }
    passSize += num;
    if (num == 0) break;
    {
      REGISTER_TIMER("TrainBatchFullGrad");
      trainerInternal_->calcGradOneBatch(batchId, dataBatch);
    }
    ++batchId;
  }

  trainerInternal_->getGradientMachine()->onPassEnd();
  // actually, at the end of pass, aggregate gradients
  trainerInternal_->getParameterUpdater()->finishBatch(0);

  LOG(INFO) << "Calc Full Gradient: "
            << " Pass=" << passId
            << " " << stats_->getStats(false /*without current cost*/);
}

void TrainerVR::train(size_t numPasses) {
  srand(config_->getConfig().start_pass() + 1);
  dataProvider_->reset();

  if (this->testDataProvider_) {
    this->testDataProvider_->reset();
  }

  trainerInternal_->getGradientMachine()->start(*config_, dataProvider_);

  // init w_s = w,
  trainerInternal_->copyToSnapshotParameter();
  for (size_t i = 0; i < numPasses; ++i) {
    // P_GRAD_SUM = 0
    trainerInternal_->clearGradients(PARAMETER_GRADIENT_SUM);
    // P_GRAD = 0
    trainerInternal_->clearGradients(PARAMETER_GRADIENT);
    // full gradient g = \sum_b \partial f_b(w_s)
    calculateFullGradient(config_->getConfig().start_pass() + i);
    dataProvider_->setSkipShuffle();
    dataProvider_->reset();

    // P_GRAD = 0
    trainerInternal_->clearGradients(PARAMETER_GRADIENT);
    // w_t = w_s, t=0
    trainerInternal_->copyFromSnapshotParameter();
    // for each batch, g_{t+1} = \partial f_b(w_t) - \partial f_b(w_s) + g
    // w_{t+1} = w_t - \eta g_{t+1}
    trainOnePass(config_->getConfig().start_pass() + i);
    if (i < numPasses - 1) {
      dataProvider_->reset();
    }
    // w_s = w_T, P_VALUE -> P_SNAPSHOT_VALUE
    trainerInternal_->copyToSnapshotParameter();
  }

  trainerInternal_->getGradientMachine()->finish();
}

void TrainerVR::trainOnePass(int passId) {
  this->stats_->reset();
  int64_t batchId = 0;
  int32_t batchSize = config_->getOptConfig().batch_size();

  trainerInternal_->getParameterUpdater()->startPass();
  evaluator_->start();
  while (true) {
    DataBatch dataBatch;

    int num = 0;
    {
      REGISTER_TIMER("getTrainBatch");
      num = dataProvider_->getNextBatch(batchSize, &dataBatch);
    }
    if (num == 0) break;
    {
      REGISTER_TIMER("TrainBatch");
      trainerInternal_->trainOneBatch(batchId, dataBatch);
    }

    ++batchId;
  }

  if (batchId == 0) {
    // This means no more data from DataProvider
    return;
  }

  trainerInternal_->finishTrainPass(passId, batchId);

  FOR_TIMING(globalStat.setThreadInfo(true));
  FOR_TIMING(globalStat.printAllStatus());
  FOR_TIMING(globalStat.reset());

  if (testDataProvider_) {
    tester_->testOnePeriod();
  }

  if (passId % FLAGS_saving_period == 0 && FLAGS_trainer_id == 0) {
    paramUtil_->saveParametersOnePass(passId);
  }
}

}  // namespace paddle
