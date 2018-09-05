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

#include <thread>
#include "paddle/pserver/ParameterClient2.h"
#include "ParameterUpdater.h"
#include "paddle/utils/Queue.h"
#include "RemoteParameterUpdater.h"

namespace paddle {

class VRRemoteParameterUpdater : public RemoteParameterUpdater {
public:
  VRRemoteParameterUpdater(OptimizationConfig config, int expectedPassCount);

  virtual void init(std::vector<ParameterPtr>& parameters);
  virtual void finishBatch(real cost);
  virtual void startPass();
  virtual bool finishPass(real cost);
protected:
  virtual void controller();
};

}  // namespace paddle
