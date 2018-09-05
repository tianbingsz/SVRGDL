/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "ImageExtract.h"
#include "paddle/utils/Locks.h"
#include "paddle/utils/ThreadLocal.h"
#include "paddle/utils/TypeDefs.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {

class ImageExtract;
typedef std::shared_ptr<ImageExtract> ImageExtractPtr;

// data subtracted mean value
class ImageDataProvider : public DataProvider {
protected:
  ImageExtractPtr imgData_;
  std::mutex lock_;

public:
  ImageDataProvider(const DataConfig& config, bool useGpu);
  ~ImageDataProvider();
  int64_t getNextBatchInternal(int64_t size, DataBatch* batch);
  virtual void reset();
  virtual int64_t getSize();
  virtual void shuffle() {}
};

}  // namespace paddle
