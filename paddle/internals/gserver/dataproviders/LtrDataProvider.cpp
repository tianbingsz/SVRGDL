/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "LtrDataProvider.h"
#include "paddle/utils/Util.h"
#include "paddle/utils/Logging.h"
#include "paddle/gserver/dataproviders/DataProviderGroup.h"
#include "paddle/utils/StringUtil.h"
#include <algorithm>

namespace paddle {

using namespace std;

REGISTER_DATA_PROVIDER(ltr, LtrDataProvider);
REGISTER_DATA_PROVIDER(ltr_sparse_sequence, LtrSparseSequenceDataProvider);
REGISTER_DATA_PROVIDER(ltr_sparse, LtrSparseDataProvider<sparse_non_value_t>);
REGISTER_DATA_PROVIDER(ltr_sparse_value,
                       LtrSparseDataProvider<sparse_float_value_t>);
REGISTER_DATA_PROVIDER(
    ltr_group, DataProviderGroup<LtrSparseDataProvider<sparse_non_value_t>>);
REGISTER_DATA_PROVIDER(ltr_sparse_sequence_group,
                       DataProviderGroup<LtrSparseSequenceDataProvider>);

template <class T>
LtrSparseDataProvider<T>::LtrSparseDataProvider(const DataConfig& config,
                                                bool useGpu, bool loadDataAll)
    : DataProvider(config, useGpu) {
  /* initialize the size of a sample, and the buffer */
  nextItemIndex_ = 0;

  size_t slot_num = config_.slot_dims_size();
  slots_.resize(slot_num);
  for (size_t i = 0; i < slots_.size(); ++i) {
    slots_[i].sampleDim = config_.slot_dims(i);
    CHECK_GT(slots_[i].sampleDim, 0);
  }

  size_t float_slot_num = config_.float_slot_dims_size();
  floatSlots_.resize(float_slot_num);
  for (size_t i = 0; i < floatSlots_.size(); ++i) {
    floatSlots_[i].sampleDim = config_.float_slot_dims(i);
    CHECK_GT(floatSlots_[i].sampleDim, 0);
  }
  if (loadDataAll) {
    loadData(config_.files());
  }
}

template <class T>
void LtrSparseDataProvider<T>::shuffle() {
  struct PairStruct {
    PairStruct(int _first, int _second, real _label)
        : first(_first), second(_second), label(_label) {}
    int first, second;
    real label;
  };

  std::vector<PairStruct> pairVec;
  size_t len = pairLabels_.size();
  for (size_t i = 0; i < len; i++) {
    pairVec.push_back(PairStruct(pairs_[0][i], pairs_[1][i], pairLabels_[i]));
  }
  std::random_shuffle(pairVec.begin(), pairVec.end());
  pairs_[0].clear();
  pairs_[1].clear();
  pairLabels_.clear();
  for (auto pair : pairVec) {
    pairs_[0].push_back(pair.first);
    pairs_[1].push_back(pair.second);
    pairLabels_.push_back(pair.label);
  }
}

template <class T>
int64_t LtrSparseDataProvider<T>::getNextBatchInternal(int64_t size,
                                                       DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  std::lock_guard<RWLock> guard(lock_);
  startIndex = nextItemIndex_;
  cpySize = std::min(size, this->getSize() - nextItemIndex_);
  nextItemIndex_ += cpySize;

  if (cpySize > 0) {
    for (unsigned int k = 0; k < 2; ++k) {
      auto& dataBatchs = *(dataBatchs_[k]);
      if (dataBatchs.size() < slots_.size()) {
        dataBatchs.resize(slots_.size());
      }
      auto& floatDataBatchs = *(floatDataBatchs_[k]);
      if (floatDataBatchs.size() < floatSlots_.size()) {
        floatDataBatchs.resize(floatSlots_.size());
      }
      for (unsigned int i = 0; i < slots_.size(); ++i) {
        auto& slot = slots_[i];
        T* data = slot.data.data();
        int64_t* indexs = slot.indexs.data();
        int64_t* seqs = pairs_[k].data() + startIndex;

        /* samples data */
        auto& dataBatch = dataBatchs[i];
        if (!dataBatch) {
          dataBatch = Matrix::createSparseMatrix(
              cpySize, slot.sampleDim,
              cpySize * 20, /* DEFAULT_AVG_WIDTH = 20 */
              sizeof(T) == sizeof(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE,
              SPARSE_CSR,
              /* trans= */ false, useGpu_);
        } else {
          dataBatch->resize(cpySize, slot.sampleDim);
        }

        if (std::dynamic_pointer_cast<GpuSparseMatrix>(dataBatch)) {
          (std::dynamic_pointer_cast<GpuSparseMatrix>(dataBatch))
              ->copyFrom(seqs, indexs, data, HPPL_STREAM_DEFAULT);
        } else if (std::dynamic_pointer_cast<CpuSparseMatrix>(dataBatch)) {
          (std::dynamic_pointer_cast<CpuSparseMatrix>(dataBatch))
              ->copyFrom(seqs, indexs, data);
        } else {
          LOG(FATAL) << "Not supported";
        }
        batch->appendData(dataBatch);
      }
      // float slots
      for (unsigned int i = 0; i < floatSlots_.size(); ++i) {
        auto& slot = floatSlots_[i];
        real* data = slot.data.data();
        int64_t* seqs = pairs_[k].data() + startIndex;
        auto& dataBatch = floatDataBatchs[i];
        Matrix::resizeOrCreate(dataBatch, cpySize, slot.sampleDim);
        dataBatch->copyFrom(data, seqs);
        batch->appendData(dataBatch);
      }
    }

    real* label = pairLabels_.data() + startIndex;
    MatrixPtr& labelBatch = *labelBatch_;  // get the thread local object
    Matrix::resizeOrCreate(labelBatch, cpySize, /*  width= */ 1,
                           /*  trans= */ false, useGpu_);
    labelBatch->copyFrom(label, cpySize * 1);
    batch->appendData(labelBatch);
  }
  batch->setSize(cpySize);
  return cpySize;
}

template <class T>
void LtrSparseDataProvider<T>::reset() {
  nextItemIndex_ = 0;
  if (!skipShuffle_) {
    shuffle();
  }
  DataProvider::reset();
}

template <class T>
int64_t LtrSparseDataProvider<T>::getSize() {
  int64_t size = pairLabels_.size();
  if (usageRatio_ < 1.0f) {
    size = static_cast<int64_t>(size * usageRatio_);
  }
  return size;
}

const int kMaxNumOfFea = 10000;

template <class T>
void LtrSparseDataProvider<T>::loadFloatFea(FILE* fp, FloatSlot& slot) {
  size_t currentSize = slot.data.size();
  size_t appendSize = slot.sampleDim;
  slot.data.resize(currentSize + appendSize);
  uint16_t temp;
  CHECK_EQ(1U, fread(&temp, sizeof(temp), 1, fp));  // skip slot id
  CHECK_EQ(1U, fread(&temp, sizeof(temp), 1, fp));  // num of fea
  CHECK_EQ(temp, appendSize);
  CHECK_EQ(appendSize,
           fread(slot.data.data() + currentSize, sizeof(real), appendSize, fp));
}

template <class T>
void LtrSparseDataProvider<T>::loadFea(FILE* fp, Slot& slot) {
  unsigned short numoffea;
  CHECK_EQ((size_t)1, fread(&numoffea, sizeof(short), 1, fp));
  CHECK_LE(numoffea, (size_t)kMaxNumOfFea);

  slot.data.resize(slot.indexs.back() + numoffea);
  CHECK_EQ(numoffea, fread(slot.data.data() + slot.indexs.back(), sizeof(T),
                           numoffea, fp));
  for (int64_t i = slot.indexs.back(); i < slot.indexs.back() + numoffea; ++i) {
    CHECK_LT(slot.data[i].col, (size_t)slot.sampleDim)
        << " fea " << slot.data[i].col << " id wrong, index:" << i;
  }
  slot.indexs.push_back(slot.indexs.back() + numoffea);
}

template <class T>
size_t LtrSparseDataProvider<T>::loadDataFile(const string& fileName) {
  FILE* fp = fopen(fileName.c_str(), "rb");
  CHECK(fp) << " fopen " << fileName << " failed";

  size_t pos = 0;
  size_t neg = 0;

  int queryid = (infos_.size() > 0) ? (infos_.back() + 1) : 0;
  int titleid = (titleInfos_.size() > 0) ? (titleInfos_.back() + 1) : 0;
  struct sign_t {
    char data[16];
  };
  sign_t querySign;
  sign_t titleSign;
  while (!feof(fp)) {
    LabelInfo labelInfo;
    if (1 != fread(&labelInfo, sizeof(labelInfo), 1, fp)) {
      CHECK(feof(fp)) << " fread label failed ";
      break;
    }
    CHECK_GT(labelInfo.numShows, 0);
    // CHECK_GE(labelInfo.numShows, labelInfo.numClicks);
    // CHECK_GT(labelInfo.var, 0.0);

    if (labelInfo.numClicks == 0) {
      neg++;
    } else {
      pos++;
    }
    labelInfos_.push_back(labelInfo);

    if (slots_.size() == 1) {
      loadFea(fp, slots_[0]);
    } else {
      unsigned short numofslot;
      CHECK_EQ((size_t)1, fread(&numofslot, sizeof(short), 1, fp));
      numofslot -= floatSlots_.size();
      // num of sparse slots: total slots - float slots

      sign_t inputSign;
      // read query sign (char [16])
      CHECK_EQ(1UL, fread(&inputSign, sizeof(inputSign), 1, fp));
      if (memcmp(&querySign, &inputSign, sizeof(inputSign)) != 0) {
        queryid++;
        querySign = inputSign;
      }
      infos_.push_back(queryid);

      // read url sign (char [16])
      CHECK_EQ(1UL, fread(&inputSign, sizeof(inputSign), 1, fp));
      if (memcmp(&titleSign, &inputSign, sizeof(inputSign)) != 0) {
        titleid++;
        titleSign = inputSign;
      }
      titleInfos_.push_back(titleid);

      short slotIdLast = -1;
      for (size_t i = 0; i < numofslot; ++i) {
        short slotId;
        CHECK_EQ((size_t)1, fread(&slotId, sizeof(short), 1, fp));
        slotId--;  // start from 1 -> from 0
        CHECK_GT(slotId, slotIdLast);
        for (int j = slotIdLast + 1;
             j < std::min((int)slotId, (int)slots_.size()); j++) {
          slots_[j].indexs.push_back(slots_[j].indexs.back());
        }

        if ((size_t)slotId < slots_.size()) {
          loadFea(fp, slots_[slotId]);
        } else {  // drop slot
          unsigned short numoffea;
          CHECK_EQ((size_t)1, fread(&numoffea, sizeof(short), 1, fp));
          CHECK_LE(numoffea, (size_t)kMaxNumOfFea);

          T data[kMaxNumOfFea];
          CHECK_EQ(numoffea, fread(data, sizeof(T), numoffea, fp));
        }
        slotIdLast = slotId;
      }
      for (size_t j = slotIdLast + 1; j < slots_.size(); ++j) {
        slots_[j].indexs.push_back(slots_[j].indexs.back());
      }

      // debug check
      for (auto& slot : slots_) {
        CHECK_EQ(slot.indexs.size() - 1, labelInfos_.size());
      }
    }
    // for float slots
    for (size_t i = 0; i < floatSlots_.size(); i++) {
      loadFloatFea(fp, floatSlots_[i]);
    }
  }
  fclose(fp);

  return pos;
}

template <class T>
void LtrSparseDataProvider<T>::loadData(const string& fileListFileName) {
  std::vector<std::string> fileList;
  loadFileList(fileListFileName, fileList);
  loadData(fileList);
}

template <class T>
void LtrSparseDataProvider<T>::loadData(
    const std::vector<std::string>& fileList) {
  // clear data
  labelInfos_.clear();
  infos_.clear();
  for (auto& slot : slots_) {
    slot.indexs.clear();
    slot.data.clear();
  }
  pairs_[0].clear();
  pairs_[1].clear();
  pairLabels_.clear();

  // init slots
  for (auto& slot : slots_) {
    slot.indexs.push_back(0);
  }

  size_t pos = 0;
  for (auto& file : fileList) {
    LOG(INFO) << "load data file " << file;
    pos += loadDataFile(file);
  }

  // stats
  size_t dataSize = 0;
  for (auto& slot : slots_) {
    CHECK_EQ(slot.indexs.size() - 1, labelInfos_.size());
    dataSize += slot.data.size();
  }

  auto fillPairFunc = [&](size_t first, size_t second) {
    CHECK_NE(first, second);
    // skip same url pair
    if (titleInfos_[first] == titleInfos_[second]) {
      VLOG(2) << "url same: " << first << " vs " << second;
      return false;
    }
    /*
    LabelInfo& infoFirst = labelInfos_[first];
    LabelInfo& infoSecond = labelInfos_[second];
    float x = (infoFirst.mean - infoSecond.mean) /
    std::sqrt(infoFirst.var + infoSecond.var);
    float phi = (1 + std::erf(x / std::sqrt(2.))) / 2;

    if (phi < config_.ltr_conf().pair_filter_phi_threshold()) { return false; }
    */
    pairs_[0].push_back(first);
    pairs_[1].push_back(second);
    pairLabels_.push_back(1.0f);  //(phi);//

    return true;
  };

  // fill pairs
  vector<size_t> zeroVec;
  vector<size_t> oneVec;
  auto fillDataFunc = [&](size_t start, size_t end) {
    zeroVec.clear();
    oneVec.clear();
    for (size_t i = start; i < end; ++i) {
      if (labelInfos_[i].numClicks) {
        oneVec.push_back(i);
      } else {
        zeroVec.push_back(i);
      }
    }
    if (zeroVec.empty() || oneVec.empty()) {
      return;
    }
    uint32_t maxNumPair = config_.ltr_conf().max_num_pair_per_query();
    if (0 == maxNumPair) {
      // make ALL possible pairs
      for (size_t i = 0; i < oneVec.size(); ++i) {
        for (size_t j = 0; j < zeroVec.size(); ++j) {
          fillPairFunc(oneVec[i], zeroVec[j]);
        }
      }
    } else {
      std::random_shuffle(zeroVec.begin(), zeroVec.end());
      std::random_shuffle(oneVec.begin(), oneVec.end());
      size_t len = std::min(zeroVec.size(), (size_t)maxNumPair);
      for (size_t i = 0; i < len; ++i) {
        size_t first = oneVec[i % oneVec.size()];
        size_t second = zeroVec[i];
        fillPairFunc(first, second);
      }
    }
  };

  auto start = infos_.begin();
  while (start != infos_.end()) {
    auto end = std::find_if(start + 1, infos_.end(),
                            [=](int x) { return x != *start; });
    CHECK(end != start);
    fillDataFunc(start - infos_.begin(), end - infos_.begin());

    start = end;
  }

  double phiTotal = 0;
  for (auto phi : pairLabels_) {
    CHECK(phi <= 1.0 && phi >= 0.0);
    phiTotal += phi;
  }
  double phiAvg = phiTotal / pairLabels_.size();

  LOG(INFO) << "read done, num of instance=" << labelInfos_.size()
            << " num of positive instance=" << pos
            << " num of slot=" << slots_.size()
            << " num of float slot=" << floatSlots_.size()
            << " num of pairs=" << pairs_[0].size() << " avg of phi=" << phiAvg
            << " num of query=" << infos_.back() << " data size=" << dataSize;
}

template class LtrSparseDataProvider<sparse_non_value_t>;
template class LtrSparseDataProvider<sparse_float_value_t>;

LtrSparseSequenceDataProvider::LtrSparseSequenceDataProvider(
    const DataConfig& config, bool useGpu, bool loadDataAll)
    : LtrSparseDataProvider<sparse_non_value_t>(config, useGpu, loadDataAll) {
  CHECK(!useGpu) << "LtrSparseSequenceDataProvider does not support gpu";
}

int64_t LtrSparseSequenceDataProvider::getNextBatchInternal(int64_t size,
                                                            DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  // cpySize: how many pairs returned
  std::lock_guard<RWLock> guard(lock_);
  startIndex = nextItemIndex_;
  cpySize = std::min(size, this->getSize() - nextItemIndex_);
  nextItemIndex_ += cpySize;

  // for sequence, similiar to ProtoDataProvider
  DataBatch& cpuBatch = *cpuBatch_;
  std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
  cpuBatch.setSize(cpySize);
  cpuArguments.resize(slots_.size() * 2 + 1);  // pair and label

  int slotNum = slots_.size();

  if (cpySize > 0) {
    for (int pair = 0; pair < 2; pair++) {
      // pair=0: left, pair=1: right
      for (size_t slot = 0; slot < slots_.size(); slot++) {
        // pointers used in current slot
        auto& thisslot = slots_[slot];
        sparse_non_value_t* data = thisslot.data.data();
        int64_t* indexs = thisslot.indexs.data();
        int64_t* seqs = pairs_[pair].data() + startIndex;

        // current slot: i need cpySize instances. what is the total length?
        int totalFeatureInCurrentSlot = 0;
        for (int ins = 0; ins < cpySize; ins++) {
          int64_t currInsId = seqs[ins];
          totalFeatureInCurrentSlot +=
              indexs[currInsId + 1] - indexs[currInsId];
          // special: if current instance has NO feature in current slot
          if (indexs[currInsId + 1] - indexs[currInsId] == 0) {
            totalFeatureInCurrentSlot++;
          }
        }
        // done

        // current slot: sequenceStartPositions
        ICpuGpuVector::resizeOrCreate(
            cpuArguments[slot + slotNum * pair].sequenceStartPositions,
            cpySize + 1, /* useGpu= */ false);

        // current slot: ids
        IVector::resizeOrCreate(cpuArguments[slot + slotNum * pair].ids,
                                totalFeatureInCurrentSlot,
                                /* useGpu= */ false);

        // where to write
        int* currPosOfArgumentId =
            cpuArguments[slot + slotNum * pair].ids->getData();
        int* currPosOfArgumentSeqStart =
            cpuArguments[slot + slotNum * pair].sequenceStartPositions
            ->getMutableData(false);
        int allSequenceLength = 0;
        currPosOfArgumentSeqStart[0] = 0;
        // for each instance, copy data and fill sequence positions
        for (int instance = 0; instance < cpySize; instance++) {
          int64_t currInsId = pairs_[pair][startIndex + instance];
          int64_t currInsLength = indexs[currInsId + 1] - indexs[currInsId];
          sparse_non_value_t* currInstanceData = data + indexs[currInsId];
          // write sequenceStartPositions
          allSequenceLength += currInsLength;
          currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
          // copy features
          for (int featCopier = 0; featCopier < currInsLength; featCopier++) {
            currPosOfArgumentId[featCopier] = currInstanceData[featCopier].col;
          }
          currPosOfArgumentId += currInsLength;
          // special: if current instance has NO feature in current slot
          if (currInsLength == 0) {
            allSequenceLength++;
            currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
            currPosOfArgumentId[0] = -1;
            currPosOfArgumentId++;
          }
          // done
        }  // end for every instance
      }    // end for slot
    }      // end for pair

    // label
    Matrix::resizeOrCreate(cpuArguments[slots_.size() * 2].value, cpySize, 1,
                           /*  trans */ false, /*  useGpu */ false);
    cpuArguments[slots_.size() * 2].value->copyFrom(
        pairLabels_.data() + startIndex, cpySize);
    // label HAS sequence structure
    ICpuGpuVector::resizeOrCreate(
        cpuArguments[slots_.size() * 2].sequenceStartPositions, cpySize + 1,
        /* useGpu */ false);
    cpuArguments[slots_.size() * 2].sequenceStartPositions->fillSequence(false);
  }
  *batch = cpuBatch;
  batch->setSize(cpySize);
  return cpySize;
}

LtrDataProviderBase::LtrDataProviderBase(const DataConfig& config, bool useGpu,
                                         bool withInfo)
    : DataProvider(config, useGpu) {
  /* initialize the size of a sample, and the buffer */
  sampleDim_ = config_.feat_dim() * (2 * config_.context_len() + 1);
  bufferCapacity_ = config_.buffer_capacity();
  withInfo_ = withInfo;
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;

  /* malloc buffer in cpu */
  for (size_t i = 0; i < 2; ++i) {
    hInputDataBuf_[i] = make_shared<CpuMatrix>(bufferCapacity_, sampleDim_);
  }
  hInputCtrBuf_ = make_shared<CpuMatrix>(bufferCapacity_, 1);
  hInputLabelBuf_ = make_shared<CpuIVector>(bufferCapacity_);
  hInputInfoBuf_ = make_shared<CpuIVector>(bufferCapacity_);
}

void LtrDataProviderBase::shuffle() {
  int i, t;
  int len = sampleNumInBuf_;
  std::vector<real> temp(sampleDim_);
  real* ctr = hInputCtrBuf_->getData();
  int* label = hInputLabelBuf_->getData();
  int* info = hInputInfoBuf_->getData();
  int sampleSz = sizeof(real) * sampleDim_;
  for (i = 0; i < len; i++) {
    int randNum = rand();
    t = randNum % (len - i) + i;
    // swap
    if (i != t) {
      // swap data
      for (size_t j = 0; j < 2; ++j) {
        real* data = hInputDataBuf_[j]->getData();

        memcpy(&temp[0], &data[i * sampleDim_], sampleSz);
        memcpy(&data[i * sampleDim_], &data[t * sampleDim_], sampleSz);
        memcpy(&data[t * sampleDim_], &temp[0], sampleSz);
      }

      std::swap(ctr[i], ctr[t]);
      std::swap(label[i], label[t]);
      if (withInfo_) {
        std::swap(info[i], info[t]);
      }
    }
  }
}

int64_t LtrDataProviderBase::getNextBatchInternal(int64_t size,
                                                  DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  std::lock_guard<RWLock> guard(lock_);
  if (sampleNumInBuf_ - nextItemIndex_ < size) {
    int64_t n = fillBuffer();
    LOG(INFO) << "fillBuffer return " << n << " samples.\n";
  }
  startIndex = nextItemIndex_;
  cpySize = std::min(size, sampleNumInBuf_ - nextItemIndex_);
  nextItemIndex_ += cpySize;

  if (cpySize > 0) {
    for (size_t i = 0; i < 2; ++i) {
      real* data = hInputDataBuf_[i]->getData() + startIndex * sampleDim_;
      MatrixPtr& dataBatch = *(dataBatch_[i]);  // get the thread local object
      Matrix::resizeOrCreate(dataBatch, cpySize, sampleDim_, false, useGpu_);
      dataBatch->copyFrom(data, cpySize * sampleDim_);
      batch->appendData(dataBatch);
    }

    real* ctr = hInputCtrBuf_->getData() + startIndex;
    int* label = hInputLabelBuf_->getData() + startIndex;
    int* info = hInputInfoBuf_->getData() + startIndex;

    MatrixPtr& ctrBatch = *ctrBatch_;       // get the thread local object
    IVectorPtr& labelBatch = *labelBatch_;  // get the thread local object
    IVectorPtr& infoBatch = *infoBatch_;    // get the thread local object

    Matrix::resizeOrCreate(ctrBatch, cpySize, 1, false, useGpu_);
    ctrBatch->copyFrom(ctr, cpySize * 1);
    batch->appendData(ctrBatch);

    IVector::resizeOrCreate(labelBatch, cpySize, useGpu_);
    batch->appendLabel(labelBatch);
    labelBatch->copyFrom(label, cpySize);

    if (withInfo_) {
      IVector::resizeOrCreate(infoBatch, cpySize, useGpu_);
      infoBatch->copyFrom(info, cpySize);
      batch->appendLabel(infoBatch);
    }
  }
  batch->setSize(cpySize);
  return cpySize;
}

void LtrDataProviderBase::reset() {
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;
  DataProvider::reset();
}

int64_t LtrDataProviderBase::getSize() {
  LOG(FATAL) << "Currently, not implemented";
  return 0;
}

int64_t LtrDataProviderBase::fillBuffer() {
  int64_t n = sampleNumInBuf_ - nextItemIndex_;

  /* flash the remaining data to the beginning of the buffer */
  if (n > 0) {
    for (size_t i = 0; i < 2; ++i) {
      hInputDataBuf_[i]->copyFrom(
          hInputDataBuf_[i]->getData() + nextItemIndex_ * sampleDim_,
          n * sampleDim_);
    }
    hInputCtrBuf_->copyFrom(hInputCtrBuf_->getData() + nextItemIndex_, n);
    hInputLabelBuf_->copyFrom(hInputLabelBuf_->getData() + nextItemIndex_, n);
    if (withInfo_) {
      hInputInfoBuf_->copyFrom(hInputInfoBuf_->getData() + nextItemIndex_, n);
    }
  }
  real* data[2];
  for (size_t i = 0; i < 2; ++i) {
    data[i] = hInputDataBuf_[i]->getData() + n * sampleDim_;
  }
  sampleNumInBuf_ =
      n + fillBufferImp(data, hInputCtrBuf_->getData() + n,
                        hInputLabelBuf_->getData() + n,
                        hInputInfoBuf_->getData() + n, bufferCapacity_ - n);

  /* for stachastic gradient training */
  if (!skipShuffle_) {
    shuffle();
  }

  nextItemIndex_ = 0;

  return sampleNumInBuf_;
}

LtrDataProvider::LtrDataProvider(const DataConfig& config, bool useGpu)
    : LtrDataProviderBase(config, useGpu, true /*with info*/),
      currentSampleIndex_(0) {
  loadData(config_.files());
}

LtrDataProvider::~LtrDataProvider() {}

int64_t LtrDataProvider::fillBufferImp(real* data[], real* ctr, int* label,
                                       int* info, int64_t size) {
  (void)ctr;
  const size_t MAX_PAIRS_PER_QUERY = 500000;
  size_t fillPos = 0;

  // fill pairs
  auto fillDataFunc = [&](size_t start, size_t end) {
    size_t ones = std::count(labels_.data() + start, labels_.data() + end, 1);
    size_t randSelect =
        skipShuffle_ ? 0 : ones;  // skip randselect for test pass

    size_t counter = 0;
    for (size_t i = start; i < end; ++i) {
      for (size_t j = i + 1; j < end; ++j) {
        if (labels_[i] == labels_[j]) {
          continue;
        }
        if (randSelect > 1) {
          // rand select 1/randSelect instance
          if (::rand() % randSelect != 0) {
            continue;
          }
        }
        size_t first = labels_[i] > labels_[j] ? i : j;
        size_t second = labels_[i] > labels_[j] ? j : i;
        memcpy(data[0] + fillPos * sampleDim_, &data_[first * sampleDim_],
               sampleDim_ * sizeof(real));
        memcpy(data[1] + fillPos * sampleDim_, &data_[second * sampleDim_],
               sampleDim_ * sizeof(real));
        label[fillPos] = 1;
        info[fillPos] = infos_[first];
        fillPos++;

        counter++;
        CHECK_LE(counter, MAX_PAIRS_PER_QUERY);
      }
    }
    if (counter > 10000) {
      LOG(INFO) << "queryid " << infos_[start]
                << " to large, has ins=" << end - start
                << " pairs= " << counter;
    }
  };

  auto start = infos_.begin() + currentSampleIndex_;
  while (start != infos_.end() &&
         fillPos + MAX_PAIRS_PER_QUERY < (size_t)size) {
    auto end = std::find_if(start + 1, infos_.end(),
                            [=](int x) { return x != *start; });
    CHECK(end != start);
    fillDataFunc(start - infos_.begin(), end - infos_.begin());
    currentSampleIndex_ = end - infos_.begin();

    start = end;
  }

  return fillPos;
}

void LtrDataProvider::reset() {
  currentSampleIndex_ = 0;
  LtrDataProviderBase::reset();
}

void LtrDataProvider::loadData(const string& fileListFileName) {
  std::vector<std::string> fileList;
  loadFileList(fileListFileName, fileList);

  for (auto& file : fileList) {
    LOG(INFO) << "load data file " << file;
    loadDataFile(file);
  }

  LOG(INFO) << "read done, num of instance=" << labels_.size()
            << " data size=" << data_.size();
}

void LtrDataProvider::loadDataFile(const string& fileName) {
  ifstream is(fileName);
  std::string line;
  vector<string> pieces;
  while (is) {
    if (!getline(is, line)) break;
    str::split(line, '\t', &pieces);
    CHECK_EQ((size_t)(sampleDim_ + 2), pieces.size())
        << " Dimension mismatch, " << pieces.size() - 2 << " in " << fileName
        << " " << sampleDim_ << " from config";
    labels_.push_back(atoi(pieces[0].c_str()));
    uint64_t queryid = strtoul(pieces[1].c_str(), NULL, 0);
    infos_.push_back(static_cast<int>(queryid & 0x1fffffff));
    for (int i = 0; i < sampleDim_; ++i) {
      data_.push_back(atof(pieces[i + 2].c_str()));
    }
  }
}

}  // namespace paddle
