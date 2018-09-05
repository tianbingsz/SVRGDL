/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include <memory>
#include "paddle/gserver/layers/Layer.h"
#include "paddle/gserver/layers/DataLayer.h"

namespace paddle {

class DataTrimLayer : public DataLayer {
public:
  explicit DataTrimLayer(const LayerConfig& config) : DataLayer(config) {}

  void prefetch() {
    // prefetch sparse matrix/ids only
    copyDataToOutput(output_);
  }

  void copyDataToOutput(Argument& output);
};
}  // namespace paddle
