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
#include <fstream>
#include <stdlib.h>

#include "hl_gpu.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "ParameterUpdater.h"
#include "TrainerConfig.pb.h"
#include "TrainerInternal.h"
#include "TrainerConfigHelper.h"
#include "TrainerInternalConfig.h"


namespace paddle {

/**
 * TrainerInteral
 * the core training class for driving training logic
 */
class TrainerInternalVR : public TrainerInternal {
public:
  TrainerInternalVR() {
  }

  /**
   * Intializes trainer internal class
   * @param config network config
   * @param machine gradient machine
   * @param intconfig training config
   * @param stats training stats
   * @param testing if it is in testing phase
   */
  virtual void init(
            const std::shared_ptr<TrainerConfigHelper> &config,
            const GradientMachinePtr &machine,
            std::unique_ptr<TrainerInternalConfig> &&intconfig,
            const std::shared_ptr<TrainerStats> &stats,
            bool testing);

  virtual ~TrainerInternalVR() {}

  /**
   * CreateParameterUpdater
   * @param testing if it is in testing phase
   */
  virtual void createParameterUpdater(bool testing);

  /**
   * trainOneBatch
   * @param batchId current batch id
   * @param dataBatch data for the batch
   */
  virtual void trainOneBatch(int64_t batchId, const DataBatch& dataBatch);

  /**
   * calcGradOneBatch
   * @param batchId current batch id
   * @param dataBatch data for the batch
   */
  virtual void calcGradOneBatch(int64_t batchId, const DataBatch& dataBatch);

  /**
   * showParameterStats
   * @param paraStats training stats
   */
  virtual void showParameterStats(const std::vector<ParaStat>& paraStats);

  /**
   * swapParameter, P_VALUE <-> P_SNAPSHOT_VALUE
   *
   */
  void swapParameter();

  /**
   * copyToSnapshotParameter, P_VALUE -> P_SNAPSHOT_VALUE
   *
   */
  void copyToSnapshotParameter() {
    copyParameter(PARAMETER_VALUE, PARAMETER_SNAPSHOT_VALUE);
  }

  /**
   * copyFromSnapshotParameter, P_SNAPSHOT_VALUE -> P_VALUE
   *
   */
  void copyFromSnapshotParameter() {
    copyParameter(PARAMETER_SNAPSHOT_VALUE, PARAMETER_VALUE);
  }

  /**
   * negGradients, g = -g
   *
   */
  void negGradients();

  /**
   * clearGradients,  g = 0
   * @param parameterType P_GRAD or P_SUM_GRAD
   */
  void clearGradients(ParameterType parameterType);

protected:
  /**
   * copyParameter
   * @param fromType copy from
   * @param toType copy to
   */
  void copyParameter(ParameterType fromType, ParameterType toType);
};

}  // namespace paddle
