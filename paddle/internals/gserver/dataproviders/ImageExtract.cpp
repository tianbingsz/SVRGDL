/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "paddle/utils/PythonUtil.h"

// Do not treat the unused static function in 'arrayobject.h' as error
#pragma GCC diagnostic ignored "-Wunused-function"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <algorithm>
#include <stdlib.h>
#include "paddle/utils/Logging.h"

#include "ImageExtract.h"
namespace paddle {

void ImageExtract::init(int imgSize, int cropBorder, int channels,
                        const std::string& fileList,
                        const std::string& metaFile, const std::string& module,
                        const std::string& dataFunc,
                        const std::string& metaFunc, PassType passType,
                        bool useGpu) {
  useGpu_ = useGpu;
  imageSize_ = imgSize;
  cropBorder_ = cropBorder;
  channels_ = channels;
  int innerSize = imageSize_ - 2 * cropBorder;
  sampleDim_ = innerSize * innerSize * channels_;
  fullSize_ = imageSize_ * imageSize_ * channels_;

  module_ = module;
  dataFunc_ = dataFunc;
  metaFunc_ = metaFunc;
  passType_ = passType;

  loadFileList(fileList, files_);  // load batch names from a list
  loadMeta(metaFile);              // load meta, this is a member function

  range_.clear();
  range_.reserve(files_.size());
  int numBatches = files_.size();
  for (int i = 0; i < numBatches; ++i) {
    range_.push_back(i);
  }
}

// crop data without realloc memory
void cropDataBorder(real* data, int cpySize, int imageWidth, int channel,
                    int cropBorder, PassType passType = PASS_TRAIN) {
  CHECK(data) << "nullptr in data";
  if (cropBorder == 0) {
    return;
  }
  int dst = 0, src = 0;
  int startX = cropBorder;
  int startY = cropBorder;
  int area = imageWidth * imageWidth;
  int fullSize = area * channel;
  int innerSize = imageWidth - 2 * cropBorder;
  real* tmp = new real[innerSize];
  int isFlip = 0;
  for (int i = 0; i < cpySize; ++i) {
    if (passType == PASS_TRAIN) {
      startX = rand() %           // NOLINT
               (cropBorder * 2);  // NOLINT TODO(yuyang18): use rand_r instead.
      startY = rand() % (cropBorder * 2);  // NOLINT
      isFlip = rand() % 2;                 // NOLINT
    }
    if (!isFlip) {
      for (int c = 0; c < channel; ++c) {
        for (int y = 0; y < innerSize; ++y) {
          src = startX + (y + startY) * imageWidth + c * area + i * fullSize;
          memcpy(data + dst, data + src, sizeof(real) * innerSize);
          dst += innerSize;
        }
      }
    } else {  // horizontally flip this sample
      for (int c = 0; c < channel; ++c) {
        for (int y = 0; y < innerSize; ++y) {
          src = startX + (y + startY) * imageWidth + c * area + i * fullSize;
          // make sure no memory copy error, copy data into a buffer first
          for (int x = 0; x < innerSize; ++x) {
            tmp[innerSize - x - 1] = data[src + x];
          }
          memcpy(data + dst, tmp, sizeof(real) * innerSize);
          dst += innerSize;
        }
      }
    }
  }
  delete[] tmp;
}

int64_t ImageExtract::getNextBatch(int64_t size, DataBatch* batch) {
  if (start_ >= batchSize_) {
    // end of one batch
    if ((size_t)nextIdx_ < range_.size()) {
      // load next batch
      if (!data_.size() || !thread_) {
        // load to buffer
        bufferSize_ = pyGetNextBatch(range_[nextIdx_]);
        ++nextIdx_;
        data_.resize(buff_.size());
      } else {
        // wait buffer
        thread_->join();
        ++nextIdx_;
      }
      // swap buff_ and data_
      data_.swap(buff_);
      std::swap(batchSize_, bufferSize_);
      if (cropBorder_ > 0) {
        cropDataBorder(data_[0], batchSize_, imageSize_, channels_, cropBorder_,
                       passType_);
      }
      // asyn load to buffer
      if ((size_t)nextIdx_ < range_.size()) {
        thread_.reset(new std::thread([this]() { this->pyGetNextBatch(); }));
      } else {
        thread_ = 0;
      }
      start_ = 0;
    } else {
      // end of one epoch

      nextIdx_ = 0;
      return 0;
    }
  }
  return getNextMiniBatch(size, batch);
}

void ImageExtract::loadNextBatch() {
  // TODO(yuyang18): what is it? or just remove this data provider.
  // do nothing
}

int64_t ImageExtract::getNextMiniBatch(int64_t size, DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();
  int64_t cpySize = batchSize_ - start_ < size ? batchSize_ - start_ : size;

  // process once,
  MatrixPtr& dataBatch = *dataBatch_;
  MatrixPtr& mean = *localMean_;
  if (!mean) {
    mean = Matrix::create(1, sampleDim_, false, useGpu_);
    // copy cpu mean to gpu
    mean->copyFrom(*mean_);
  }

  if (cpySize > 0) {
    // copy data to batch
    Matrix::resizeOrCreate(dataBatch, cpySize, sampleDim_, false, useGpu_);

    dataBatch->copyFrom((real*)data_[0] + start_ * sampleDim_,
                        cpySize * sampleDim_);
    // zero data mean
    dataBatch->addBias(*mean, -1.0);  // dataBatch -= mean;

    batch->appendData(dataBatch);

    // copy label to batch
    static_assert(sizeof(real) >= sizeof(int),
                  "sizeof(real) needs to be bigger than sizeof(int)");
    IVectorPtr& labelBatchCpu = *labelBatchCpu_;
    IVectorPtr& labelBatchGpu = *labelBatchGpu_;
    IVector::resizeOrCreate(labelBatchCpu, cpySize, false);
    labelBatchCpu->copyFrom(reinterpret_cast<int*>(data_[1]) + start_, cpySize);
    if (useGpu_) {
      IVector::resizeOrCreate(labelBatchGpu, cpySize, true);
      labelBatchGpu->copyFrom(*labelBatchCpu);
      batch->appendLabel(labelBatchGpu);
    } else {
      batch->appendLabel(labelBatchCpu);
    }

    // update batch and self
    batch->setSize(cpySize);
    start_ += cpySize;
  } else {
    cpySize = 0;
  }
  return cpySize;
}

int64_t ImageExtract::pyGetNextBatch(int64_t index) {
#ifndef PADDLE_ON_LINE
  PyGuard guard;
  if (index == -1) {
    index = range_[nextIdx_];
  }
  for (size_t i = 0; i < buff_.size(); ++i) {
    if (buff_[i]) {
      delete[] buff_[i];
      buff_[i] = NULL;
    }
  }
  buff_.clear();
  PyObjectPtr ret = callPythonFuncRetPyObj(module_, dataFunc_, {files_[index]});

  // parsing data;
  PyList_Check(ret.get());
  buff_.resize(PyList_GET_SIZE(ret.get()), NULL);
  for (size_t i = 0; i < (size_t)PyList_GET_SIZE(ret.get()); ++i) {
    PyArrayObject* array = (PyArrayObject*)PyList_GET_ITEM(ret.get(), i);
    CHECK(PyArray_ISCONTIGUOUS(array));
    bufferSize_ = PyArray_DIM(array, 0);
    int nDim = PyArray_DIM(array, 1);
    buff_[i] = new real[nDim * bufferSize_];
    memcpy(buff_[i], PyArray_DATA(array), sizeof(real) * nDim * bufferSize_);
  }
#endif
  return bufferSize_;
}

void ImageExtract::loadMeta(const std::string& metaFile) {
#ifndef PADDLE_ON_LINE
  PyGuard guard;
  PyObjectPtr ret = callPythonFuncRetPyObj(module_, metaFunc_, {metaFile});
  CHECK(ret);
  PyArrayObject* array = (PyArrayObject*)(ret.get());
  int nDim = PyArray_DIM(array, 0) * PyArray_DIM(array, 1);
  CHECK_EQ(nDim, fullSize_);
  if (cropBorder_ > 0) {
    cropDataBorder((real*)PyArray_DATA(array), 1, imageSize_, channels_,
                   cropBorder_, PASS_TEST);
  }
  mean_ = Matrix::create(1, sampleDim_, false, false);
  mean_->copyFrom((real*)PyArray_DATA(array), sampleDim_);
#endif
}

void ImageExtract::reset() {
  if (passType_ == PASS_TRAIN) {
    random_shuffle(range_.begin(), range_.end());
  }
  nextIdx_ = 0;
  start_ = 0;
  batchSize_ = 0;
}

}  // namespace paddle
