/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/Util.h"
#include "paddle/utils/Logging.h"

#include <algorithm>

#include "ImageDataProvider.h"

namespace paddle {

REGISTER_DATA_PROVIDER(image, ImageDataProvider);

ImageDataProvider::ImageDataProvider(const DataConfig& config, bool useGpu)
    : DataProvider(config, useGpu) {
  CHECK_EQ(config.type(), "image");
  // train data extractor
  imgData_ = ImageExtractPtr(new ImageExtract());
  imgData_->init(
      config_.img_config().img_size(), config_.img_config().crop_size(),
      config_.img_config().channels(), config_.files().c_str(),
      config_.img_config().meta_file(), config_.img_config().module(),
      config_.img_config().data_func(), config_.img_config().meta_func(),
      config_.for_test() ? PASS_TEST : PASS_TRAIN, useGpu);
}

ImageDataProvider::~ImageDataProvider() {}

int64_t ImageDataProvider::getNextBatchInternal(int64_t size,
                                                DataBatch* batch) {
  std::lock_guard<std::mutex> guard(lock_);
  return imgData_->getNextBatch(size, batch);
}

void ImageDataProvider::reset() {
  this->imgData_->reset();
  DataProvider::reset();
}

int64_t ImageDataProvider::getSize() {
  LOG(FATAL) << "Currently, not implemented";
  return 0;
}

}  // namespace paddle
