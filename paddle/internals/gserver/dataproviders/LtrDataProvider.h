/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/gserver/dataproviders/DataProvider.h"

namespace paddle {

template <class T>
class LtrSparseDataProvider : public DataProvider {
protected:
  int64_t nextItemIndex_;  // next item to read in buffer

  ThreadLocalD<std::vector<MatrixPtr>> dataBatchs_[2];
  ThreadLocalD<std::vector<MatrixPtr>> floatDataBatchs_[2];
  ThreadLocalD<MatrixPtr> labelBatch_;

  RWLock lock_;

public:
  LtrSparseDataProvider(const DataConfig& config, bool useGpu,
                        bool loadDataAll = true);
  ~LtrSparseDataProvider() {}

  void shuffle();

  virtual void reset();

  virtual int64_t getSize();

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

  void loadData(const std::vector<std::string>& fileList);

protected:
  struct Slot {
    // indexs_.size == labels_.size + 1
    int sampleDim;  // sample feature dimension
    std::vector<int64_t> indexs;
    std::vector<T> data;
  };
  struct FloatSlot {
    // for float input
    int sampleDim;
    std::vector<real> data;
  };

  void loadData(const std::string& fileListFileName);
  size_t loadDataFile(const std::string& fileName);
  void loadFea(FILE* fp, Slot& slot);
  void loadFloatFea(FILE* fp, FloatSlot& slot);

  struct LabelInfo {
    int numShows;
    int numClicks;
    float mean;
    float var;
  };
  std::vector<LabelInfo> labelInfos_;
  std::vector<int> infos_;       // query id
  std::vector<int> titleInfos_;  // title id
  std::vector<Slot> slots_;
  std::vector<FloatSlot> floatSlots_;

  std::vector<int64_t> pairs_[2];  // rank pairs for compare
  std::vector<real> pairLabels_;   // labels(bigger or lesser) of rank pair
};

class LtrSparseSequenceDataProvider
    : public LtrSparseDataProvider<sparse_non_value_t> {
protected:
  // for sequence
  ThreadLocalD<DataBatch> cpuBatch_;

public:
  LtrSparseSequenceDataProvider(const DataConfig& config, bool useGpu,
                                bool loadDataAll = true);
  ~LtrSparseSequenceDataProvider() {}
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);
};

class LtrDataProviderBase : public DataProvider {
protected:
  int64_t sampleDim_;       // sample feature dimension
  int64_t bufferCapacity_;  // the number of samples
  int64_t sampleNumInBuf_;
  int64_t nextItemIndex_;  // next item to read in buffer
  bool withInfo_;          // some user defined info for validation

  // data buffer: bufferCapacity_ * nDataDim_
  CpuMatrixPtr hInputDataBuf_[2];
  CpuMatrixPtr hInputCtrBuf_;

  // label buffer:bufferCapacity_ * 1
  CpuIVectorPtr hInputLabelBuf_;

  // info buffer:bufferCapacity_ * 1
  CpuIVectorPtr hInputInfoBuf_;

  ThreadLocal<MatrixPtr> dataBatch_[2];
  ThreadLocal<MatrixPtr> ctrBatch_;
  ThreadLocal<IVectorPtr> labelBatch_;
  ThreadLocal<IVectorPtr> infoBatch_;

  RWLock lock_;

public:
  LtrDataProviderBase(const DataConfig& config, bool useGpu, bool withInfo);
  ~LtrDataProviderBase() {}

  void shuffle();

  virtual void reset();

  virtual int64_t getSize();

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

  // return the number of samples in the buffer
  int64_t fillBuffer();

protected:
  // Fill at most size samples into data and label
  // Each input is stored in contiguous memory locations in data.
  // data[n * sampleDim_] .. data[n * sampleDim_ + sampleDim_ - 1] is for
  // the input of the n-th sample.
  // label[n] is the label for the n-th sample.
  virtual int64_t fillBufferImp(real* data[], real* ctr, int* label, int* info,
                                int64_t size) = 0;
};

class LtrDataProvider : public LtrDataProviderBase {
public:
  LtrDataProvider(const DataConfig& config, bool useGpu);
  ~LtrDataProvider();
  virtual void reset();

protected:
  void loadData(const std::string& fileName);
  void loadDataFile(const std::string& fileName);
  virtual int64_t fillBufferImp(real* data[], real* ctr, int* label, int* info,
                                int64_t size);

protected:
  size_t currentSampleIndex_;
  std::vector<int> labels_;
  std::vector<int> infos_;  // query id
  std::vector<real> data_;
};

}  // namespace paddle
