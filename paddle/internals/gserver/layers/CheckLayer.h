/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include "paddle/gserver/layers/Layer.h"
namespace paddle {
class CheckLayer : public Layer {
public:
  explicit CheckLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  static CheckLayer* create(const LayerConfig& config);

  LayerPtr getOutputLayer() { return inputLayers_[0]; }

  LayerPtr getLabelLayer() { return inputLayers_[1]; }

  void forward(PassType passType);

  void backward(const UpdateCallback& callback = nullptr) { (void)callback; }

  void forwardImp(const MatrixPtr outputValue, Argument& label,
                  MatrixPtr result);
};

}  // namespace paddle
