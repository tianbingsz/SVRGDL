/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/gserver/layers/Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/**
 * @brief Adding random perturbation to images, including
 * flipping, rotation and scale. Only GPU is supported.
 */
class PerturbationLayer : public Layer {
public:
  explicit PerturbationLayer(const LayerConfig& config) : Layer(config) {}

  ~PerturbationLayer() {}

  /**
   * Initilize basic parent class and
   * initilize member variables from layer config.
   */
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);

protected:
  /// The patches to sample from one input image.
  int samplingRate_;
  /// The value to pad if the sampled pixel is outside of the image.
  int paddingValue_;
  /// The scale and rotation parameters.
  float scaleRatio_, rotateAngle_;
  /// The sampled patch size.
  int tgtSize_;
  /// rotation angle.
  VectorPtr rotateAngles_;
  /// scale coefficient.
  VectorPtr scales_;
  IVectorPtr centerRs_;
  IVectorPtr centerCs_;
};
}  // namespace paddle
