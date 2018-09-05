/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "CheckLayer.h"
#include "paddle/utils/Logging.h"

namespace paddle {

REGISTER_LAYER(classification_error, CheckLayer);

bool CheckLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  return Layer::init(layerMap, parameterMap);
}

void CheckLayer::forwardImp(const MatrixPtr outputValue, Argument& label,
                            MatrixPtr result) {
  CHECK_EQ(outputValue->height_, label.ids->getSize());
  result->classificationError(outputValue, label.ids);
}
void CheckLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInputValue(*getOutputLayer())->getHeight();
  int size = 1;
  resizeOutput(batchSize, size);

  MatrixPtr output = getInputValue(*getOutputLayer());
  Argument label = getInput(*getLabelLayer());

  /* get the recResult value for each sample*/
  forwardImp(output, label, this->getOutputValue());
}

}  // namespace paddle
