/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "PerturbationLayer.h"

#include "paddle/utils/Logging.h"
#include <paddle/math/Vector.h>
#include "paddle/utils/Stat.h"
#include "hl_perturbation_util.cuh"
#include "paddle/parameter/Weight.h"

namespace paddle {

REGISTER_LAYER(perturbation_layer, PerturbationLayer);

bool PerturbationLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  const PerturbationConfig& conf = config_.perturb_conf();
  samplingRate_ = conf.sampling_rate();
  paddingValue_ = conf.padding_value();
  scaleRatio_ = conf.scale();
  rotateAngle_ = conf.rotation();
  tgtSize_ = conf.target_size();
  return true;
}

void PerturbationLayer::forward(PassType passType) {
  CHECK(useGpu_) << "only gpu is supported for Perturbation Layer";
  Layer::forward(passType);

  CHECK_EQ(1U, inputLayers_.size());

  MatrixPtr inputMat = getInputValue(0);
  int dim = inputMat->getWidth();
  int batchSize = inputMat->getHeight();
  int imgSize = static_cast<int>(sqrt(dim / 3));
  resizeOutput(batchSize * samplingRate_, tgtSize_ * tgtSize_ * 3);

  real* outData = getOutputValue()->getData();
  const real* inData = inputMat->getData();
  bool isTrain = passType == PASS_TRAIN;

  Vector::resizeOrCreate(rotateAngles_, batchSize, true);
  Vector::resizeOrCreate(scales_, batchSize, true);
  IVector::resizeOrCreate(centerRs_, batchSize * samplingRate_, true);
  IVector::resizeOrCreate(centerCs_, batchSize * samplingRate_, true);

  hl_conv_random_disturb(inData, imgSize, tgtSize_, 3, batchSize, scaleRatio_,
                         rotateAngle_, samplingRate_, rotateAngles_->getData(),
                         scales_->getData(), centerRs_->getData(),
                         centerCs_->getData(), paddingValue_, isTrain, outData);
  /* activation */
  forwardActivation();
}

void PerturbationLayer::backward(const UpdateCallback& callback) {
  backwardActivation();
}

}  // namespace paddle
