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


#pragma once

#include "paddle/utils/Util.h"

#include <stdio.h>

#include "hl_gpu.h"
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"

#include "TrainerConfigHelper.h"
#include "ParameterUpdater.h"
#include "TrainerInternalVR.h"
#include "Tester.h"
#include "Trainer.h"
#include "ParamUtil.h"
#include <fstream>
#include <stdlib.h>

P_DECLARE_int32(num_passes);

namespace paddle {

/**
 * Trainer Class
 *
 * Trainer combines GradientMachine, ParameterUpdater, DataProvider together to
 * train/test a NeuralNetwork.
 */
class TrainerVR : public Trainer {
public:
  /**
   * Ctor.
   * @return
   */
  TrainerVR() {
    trainerInternal_.reset(new TrainerInternalVR());
  }

  virtual ~TrainerVR() {}

  /**
   * initialize a new trainer using config
   *
   * @param config TrainerConfig.
   * @param testing true if only for testing
   * @param gradientMachine GradientMachine that will be trained.
   *                        nullptr if create from config.
   * @param dataProvider Train Data Provider. null if create from config.
   * @param testDataProvider Test Data Provider. null if create from config.
   */
  virtual void init(
      const std::shared_ptr<TrainerConfigHelper> &config,
      bool testing = false,
      const std::shared_ptr<GradientMachine> &gradientMachine = nullptr,
      const std::shared_ptr<DataProvider> &dataProvider = nullptr,
      const std::shared_ptr<DataProvider> &testDataProvider = nullptr);

  /**
   * Train until num_passes reached.
   * One pass means neural network trains through all training data.
   *
   * @param numPasses the number of traning pass.
   * @note Durning neural network training, the num passes may set a very large
   * value, and kill training process when the result is good enough.
   */
  virtual void train(size_t numPasses = (size_t)FLAGS_num_passes);

protected:
  /**
   * Train one pass of data. passId starts from 0
   *
   * SVRG Method.
   */
  virtual void trainOnePass(int passId);

  /**
   * calculate the full gradient for a pass of data
   *
   */
  void calculateFullGradient(int passId);

protected:
  // trainer Internal
  std::unique_ptr<TrainerInternalVR> trainerInternal_;

};

}  // namespace paddle
