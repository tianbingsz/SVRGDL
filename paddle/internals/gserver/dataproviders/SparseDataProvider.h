/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/dataproviders/DataProviderGroup.h"

namespace paddle {

/**
 * Data provider for multiple sparse inputs and one integer label
 *
 * If each input element has weight, use sparse_float_value_t as T,
 * Else use sparse_non_value_t as T, which has no optional weight.
 * message element {
 *   required col uint32;//col starts from 0
 *   optional weight float;
 * }
 *
 * If fea_slot==1, data format(binary):
 * message instance {
 *   required label uint16;
 *   required numoffea uint16;
 *   repeated element feas;
 * }
 * Else, fea_slot>1, data format(binary):
 * message slot {
 *   required slotid uint16;//slotid starts from 1
 *   required numoffea uint16;
 *   repeated element feas;
 * }
 * message instance {
 *   required label uint16;
 *   required numofslot uint16;
 *   optional info char[32];
 *   repeated slot slots;
 * }
 *
 */
template <class T>
class SparseDataProvider : public DataProvider {
protected:
  int64_t nextItemIndex_;  // next item to read in buffer
  bool withInfo_;          // some user defined info for validation

  ThreadLocalD<std::vector<CpuSparseMatrixPtr>> dataBatchs_;
  ThreadLocalD<IVectorPtr> labelBatch_;
  ThreadLocalD<IVectorPtr> infoBatch_;

  // for float data
  ThreadLocalD<std::vector<MatrixPtr>> floatDataBatchs_;

  RWLock lock_;

public:
  SparseDataProvider(const DataConfig& config, bool useGpu,
                     bool loadDataAll = true);
  ~SparseDataProvider() {}

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

  void loadData(const std::string& fileName);
  size_t loadDataFile(const std::string& fileName);
  void loadFea(FILE* fp, Slot& slot);
  void loadFloatFea(FILE* fp, FloatSlot& slot);

  std::vector<int> labels_;
  std::vector<int> infos_;
  std::vector<Slot> slots_;
  std::vector<FloatSlot> floatSlots_;

  std::vector<int64_t> seqs_;   // for random shuffle
  std::vector<int> labelSeqs_;  // for random shuffle
  std::vector<int> infoSeqs_;
};

class SparseSequenceDataProvider
    : public SparseDataProvider<sparse_non_value_t> {
protected:
  // for sequence
  ThreadLocalD<DataBatch> cpuBatch_;

public:
  SparseSequenceDataProvider(const DataConfig& config, bool useGpu);
  ~SparseSequenceDataProvider() {}
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);
};

}  // namespace paddle
