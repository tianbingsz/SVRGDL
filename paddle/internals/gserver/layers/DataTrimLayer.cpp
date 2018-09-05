/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "DataTrimLayer.h"

namespace paddle {

REGISTER_LAYER(data_trim, DataTrimLayer);

void DataTrimLayer::copyDataToOutput(Argument& output) {
  CHECK_LE(getSize(), data_.value->getWidth());
  auto mat = dynamic_cast<CpuSparseMatrix*>(data_.value.get());
  if (mat == nullptr) {
    LOG(FATAL) << "TrimDataLayer only supports CpuSparseMatrix data format";
  }
  if (output.deviceId == data_.deviceId) {
    Matrix::resizeOrCreateSparseMatrix(
        output.value, mat->getHeight(), getSize(),
        mat->getHeight() /*DEFAULT_AVG_WIDTH = 1*/, mat->getValueType(),
        mat->getFormat(),
        /*trans= */ false, useGpu_);
    output.value->trimFrom(*mat);
    CHECK(!data_.grad) << "Not need gradient in DataTrimLayer";
    output.ids = data_.ids;
  } else {
    SetDevice device(output.deviceId);
    if (data_.value != nullptr) {
      Matrix::resizeOrCreateSparseMatrix(
          output.value, mat->getHeight(), getSize(),
          mat->getHeight() /*DEFAULT_AVG_WIDTH = 1*/, mat->getValueType(),
          mat->getFormat(),
          /*trans= */ false, useGpu(output.deviceId));
      output.value->trimFrom(*mat);
    }
    if (data_.grad) {
      Matrix::resizeOrCreate(output.grad, data_.value->getHeight(), getSize(),
                             /* trans= */ false, useGpu(output.deviceId));
    }
    if (data_.ids != nullptr) {
      IVector::resizeOrCreate(output.ids, data_.ids->getSize(),
                              useGpu(output.deviceId));
      output.ids->copyFrom(*data_.ids);
    }
  }
  output.setFrameHeight(data_.getFrameHeight());
  output.setFrameWidth(getSize());
  output.sequenceStartPositions = data_.sequenceStartPositions;
  output.cpuSequenceDims = data_.cpuSequenceDims;

  output.notifyValueReady();
}

}  // namespace paddle
