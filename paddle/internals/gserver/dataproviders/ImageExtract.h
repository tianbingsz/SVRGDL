/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include <vector>
#include <thread>
#include "paddle/utils/GlobalConstants.h"
#include "paddle/gserver/dataproviders/DataProvider.h"

namespace paddle {

class DataBatch;

class ImageExtract {
protected:
  int nextIdx_;  // next index in range_
  // load sequence of batches, shuffled for stochastic learning
  std::vector<int> range_;
  std::vector<real*> data_;
  std::vector<real*> buff_;
  MatrixPtr mean_;
  ThreadLocal<MatrixPtr> localMean_;
  std::vector<std::string> files_;  // training data list
  std::string module_;
  std::string dataFunc_;
  std::string metaFunc_;
  ThreadLocal<MatrixPtr> dataBatch_;
  ThreadLocal<IVectorPtr> labelBatchCpu_;
  ThreadLocal<IVectorPtr> labelBatchGpu_;
  // mini-batch start index in one batch, load new batch according
  // to the range_ when start_ is larger than batchSize_
  int64_t start_;
  int64_t batchSize_;   // batch size of current batch in memory
  int64_t bufferSize_;  // buffer size of current batch in buffer
  int64_t sampleDim_;   // sample dimension
  int64_t cropBorder_;  // crop size
  int64_t imageSize_;   // Image width and height
  int64_t channels_;    // Image Channel
  int64_t fullSize_;    // full image size = imageSize_ ** 2 * channels
  // stochastic load if passType_ == PASS_TRAIN, else sequencially load
  PassType passType_;  // data load type,
  bool useGpu_;
  // asynchronous thread for data loading
  std::unique_ptr<std::thread> thread_;

  void loadNextBatch();
  int64_t getNextMiniBatch(int64_t size, DataBatch* batch);
  void loadMeta(const std::string& metaFile);

public:
  ImageExtract() {
    nextIdx_ = 0;
    batchSize_ = 0;
    start_ = 0;
    sampleDim_ = 0;
    passType_ = PASS_TRAIN;
    thread_ = 0;
    imageSize_ = 0;
    cropBorder_ = 0;
    channels_ = 0;
  }
  ~ImageExtract() {
    for (size_t i = 0; i < data_.size(); ++i) {
      if (data_[i]) {
        delete[] data_[i];
      }
    }
    data_.clear();
    for (size_t i = 0; i < buff_.size(); ++i) {
      if (buff_[i]) {
        delete[] buff_[i];
      }
    }
    buff_.clear();
  }
  void init(int imgSize, int cropBorder, int channels,
            const std::string& fileList, const std::string& metaFile,
            const std::string& module, const std::string& dataFunc,
            const std::string& metaFunc, PassType passType, bool useGpu);

  int64_t getNextBatch(int64_t size, DataBatch* batch);
  void reset();
  int64_t pyGetNextBatch(int64_t index = -1);
};

typedef std::shared_ptr<ImageExtract> ImageExtractPtr;

}  // namespace paddle
