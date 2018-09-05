/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "SparseDataProvider.h"
#include <stdint.h>
#include <algorithm>

namespace paddle {

REGISTER_DATA_PROVIDER(sparse_non_value,
                       SparseDataProvider<sparse_non_value_t>);
REGISTER_DATA_PROVIDER(sparse_float_value,
                       SparseDataProvider<sparse_float_value_t>);
REGISTER_DATA_PROVIDER(sparse_sequence, SparseSequenceDataProvider);
REGISTER_DATA_PROVIDER(
    sparse_non_group,
    DataProviderGroup<SparseDataProvider<sparse_non_value_t>>);

template <class T>
SparseDataProvider<T>::SparseDataProvider(const DataConfig& config, bool useGpu,
                                          bool loadDataAll)
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

  withInfo_ = (slot_num > 1);
  if (loadDataAll) {
    loadData(config_.files());
  }
}

template <class T>
void SparseDataProvider<T>::shuffle() {
  std::random_shuffle(seqs_.begin(), seqs_.end());
  labelSeqs_.clear();
  for (auto& id : seqs_) {
    labelSeqs_.push_back(labels_[id]);
  }
  if (withInfo_) {
    infoSeqs_.clear();
    for (auto& id : seqs_) {
      infoSeqs_.push_back(infos_[id]);
    }
  }
}

template <class T>
int64_t SparseDataProvider<T>::getNextBatchInternal(int64_t size,
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
    auto& dataBatchs = *dataBatchs_;
    if (dataBatchs.size() == 0) {
      dataBatchs.resize(slots_.size());
    }
    auto& floatDataBatchs = *floatDataBatchs_;
    if (floatDataBatchs.size() == 0) {
      floatDataBatchs.resize(floatSlots_.size());
    }
    for (unsigned int i = 0; i < slots_.size(); ++i) {
      auto& slot = slots_[i];
      T* data = slot.data.data();
      int64_t* indexs = slot.indexs.data();
      int64_t* seqs = seqs_.data() + startIndex;

      /* samples data */
      auto& dataBatch = dataBatchs[i];
      if (!dataBatch) {
        SparseValueType valueType =
            (typeid(T) == typeid(sparse_non_value_t)) ? NO_VALUE : FLOAT_VALUE;
        dataBatch = std::make_shared<CpuSparseMatrix>(
            cpySize, slot.sampleDim, 0, valueType, SPARSE_CSR, false);
      } else {
        dataBatch->resize(cpySize, slot.sampleDim);
      }

      dataBatch->copyFrom(seqs, indexs, data);
      batch->appendData(dataBatch);
    }
    // for float slots
    for (unsigned int i = 0; i < floatSlots_.size(); ++i) {
      auto& slot = floatSlots_[i];
      real* data = slot.data.data();
      int64_t* seqs = seqs_.data() + startIndex;
      auto& dataBatch = floatDataBatchs[i];
      Matrix::resizeOrCreate(dataBatch, cpySize, slot.sampleDim);
      dataBatch->copyFrom(data, seqs);
      batch->appendData(dataBatch);
    }

    int* label =
        (labelSeqs_.size() ? labelSeqs_.data() : labels_.data()) + startIndex;
    int* info =
        (infoSeqs_.size() ? infoSeqs_.data() : infos_.data()) + startIndex;
    IVectorPtr& labelBatch = *labelBatch_;  // get the thread local object
    IVectorPtr& infoBatch = *infoBatch_;
    if (!labelBatch) {
      labelBatch = IVector::create(cpySize, FLAGS_use_gpu);
      if (withInfo_) {
        infoBatch = IVector::create(cpySize, 0);
      }
    } else {
      labelBatch->resize(cpySize);
      if (withInfo_) {
        infoBatch->resize(cpySize);
      }
    }
    labelBatch->copyFrom(label, cpySize);
    if (withInfo_) {
      infoBatch->copyFrom(info, cpySize);
    }
    batch->appendLabel(labelBatch);
    if (withInfo_) {
      batch->appendLabel(infoBatch);
    }
  }
  batch->setSize(cpySize);
  return cpySize;
}

template <class T>
void SparseDataProvider<T>::reset() {
  nextItemIndex_ = 0;
  if (!skipShuffle_) {
    shuffle();
  }
  DataProvider::reset();
}

template <class T>
int64_t SparseDataProvider<T>::getSize() {
  int64_t size = labels_.size();
  if (usageRatio_ < 1.0f) {
    size = static_cast<int64_t>(size * usageRatio_);
  }
  return size;
}

const int kMaxNumOfFea = 10000;

template <class T>
void SparseDataProvider<T>::loadFloatFea(FILE* fp, FloatSlot& slot) {
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
void SparseDataProvider<T>::loadFea(FILE* fp, Slot& slot) {
  uint16_t numoffea;
  CHECK_EQ((size_t)1, fread(&numoffea, sizeof(uint16_t), 1, fp));
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
size_t SparseDataProvider<T>::loadDataFile(const std::string& fileName) {
  FILE* fp = fopen(fileName.c_str(), "rb");
  CHECK(fp) << " fopen " << fileName << " failed";

  size_t pos = 0;
  size_t neg = 0;
  int queryid = (infos_.size() > 0) ? (infos_.back() + 1) : 0;
  std::string sign;
  struct sign_t {
    char data[16];
  };
  sign_t querySign;
  while (!feof(fp)) {
    int label;
    if (config_.read_ltr_format_input()) {
      const unsigned int LABEL_SIZE = 4;
      int labels[LABEL_SIZE];
      if (LABEL_SIZE != fread(labels, sizeof(int), LABEL_SIZE, fp)) {
        CHECK(feof(fp)) << " fread label failed ";
        break;
      }
      label = labels[1];  // click
    } else {
      uint16_t readLabel;
      if (1 != fread(&readLabel, sizeof(uint16_t), 1, fp)) {
        CHECK(feof(fp)) << " fread label failed ";
        break;
      }
      label = readLabel;
    }
    if (label == 0) {
      neg++;
    } else {
      pos++;
    }
    labels_.push_back(label);
    seqs_.push_back(seqs_.size());

    if (slots_.size() == 1) {
      loadFea(fp, slots_[0]);
    } else {
      uint16_t numofslot;
      CHECK_EQ((size_t)1, fread(&numofslot, sizeof(int16_t), 1, fp));
      numofslot -= floatSlots_.size();
      // num of sparse slots: total slots - float slots

      if (withInfo_) {
        if (config_.read_ltr_format_input()) {
          sign_t inputSign;
          CHECK_EQ(1UL, fread(&inputSign, sizeof(inputSign), 1, fp));
          if (memcmp(&querySign, &inputSign, sizeof(inputSign)) != 0) {
            queryid++;
            memcpy(&querySign, &inputSign, sizeof(inputSign));
          }
          CHECK_EQ(1UL, fread(&inputSign, sizeof(inputSign), 1, fp));
        } else {
          constexpr size_t kSignLength = 32;
          char querySign[kSignLength + 1];
          querySign[kSignLength] = '\0';
          CHECK_EQ(kSignLength,
                   fread(querySign, sizeof(char), kSignLength, fp));
          if (sign != querySign) {
            queryid++;
            sign = querySign;
          }
        }
        infos_.push_back(queryid);
      }

      int16_t slotIdLast = -1;
      for (size_t i = 0; i < numofslot; ++i) {
        int16_t slotId;
        CHECK_EQ((size_t)1, fread(&slotId, sizeof(int16_t), 1, fp));
        slotId--;  // start from 1 -> from 0
        CHECK_GT(slotId, slotIdLast);
        for (int j = slotIdLast + 1;
             j < std::min((int)slotId, (int)slots_.size()); j++) {
          slots_[j].indexs.push_back(slots_[j].indexs.back());
        }

        if ((size_t)slotId < slots_.size()) {
          loadFea(fp, slots_[slotId]);
        } else {  // drop slot
          uint16_t numoffea;
          CHECK_EQ((size_t)1, fread(&numoffea, sizeof(int16_t), 1, fp));
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
        CHECK_EQ(slot.indexs.size() - 1, labels_.size());
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
void SparseDataProvider<T>::loadData(const std::string& fileName) {
  std::vector<std::string> fileList;
  loadFileList(fileName, fileList);
  loadData(fileList);
}

template <class T>
void SparseDataProvider<T>::loadData(const std::vector<std::string>& fileList) {
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
    CHECK_EQ(slot.indexs.size() - 1, labels_.size());
    dataSize += slot.data.size();
  }

  LOG(INFO) << "read done, num of instance=" << labels_.size()
            << " num of positive instance=" << pos
            << " num of slot=" << slots_.size()
            << " num of float slot=" << floatSlots_.size()
            << " data size=" << dataSize << " slot_num=" << slots_.size();
}

template class SparseDataProvider<sparse_non_value_t>;
template class SparseDataProvider<sparse_float_value_t>;

SparseSequenceDataProvider::SparseSequenceDataProvider(const DataConfig& config,
                                                       bool useGpu)
    : SparseDataProvider<sparse_non_value_t>(config, useGpu) {
  CHECK(!useGpu) << "SparseSequenceDataProvider does not support gpu";
}

int64_t SparseSequenceDataProvider::getNextBatchInternal(int64_t size,
                                                         DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  std::lock_guard<RWLock> guard(lock_);
  startIndex = nextItemIndex_;
  cpySize = std::min(size, this->getSize() - nextItemIndex_);
  nextItemIndex_ += cpySize;

  // for sequence, as in ProtoDataProvider and LtrSparseDataProvider
  DataBatch& cpuBatch = *cpuBatch_;
  std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
  cpuBatch.setSize(cpySize);
  cpuArguments.resize(slots_.size() + 2);  // slots, label and info

  if (cpySize > 0) {
    for (size_t slot = 0; slot < slots_.size(); slot++) {
      // pointers used in current slot
      auto& thisslot = slots_[slot];
      sparse_non_value_t* data = thisslot.data.data();
      int64_t* indexs = thisslot.indexs.data();
      int64_t* seqs = seqs_.data() + startIndex;

      // current slot: i need cpySize instances. what is the total length?
      int totalFeatureInCurrentSlot = 0;
      for (int ins = 0; ins < cpySize; ins++) {
        int64_t currInsId = seqs[ins];
        totalFeatureInCurrentSlot += indexs[currInsId + 1] - indexs[currInsId];
        // special: if current instance has NO feature in current slot
        if (indexs[currInsId + 1] == indexs[currInsId]) {
          totalFeatureInCurrentSlot++;
        }
      }
      // done

      // current slot: sequenceStartPositions
      ICpuGpuVector::resizeOrCreate(
          cpuArguments[slot].sequenceStartPositions,
          cpySize + 1, /* useGpu= */ false);

      // current slot: ids
      IVector::resizeOrCreate(cpuArguments[slot].ids, totalFeatureInCurrentSlot,
                              /* useGpu= */ false);

      // where to write
      int* currPosOfArgumentId = cpuArguments[slot].ids->getData();
      int* currPosOfArgumentSeqStart =
          cpuArguments[slot].sequenceStartPositions->getMutableData(false);
      int allSequenceLength = 0;
      currPosOfArgumentSeqStart[0] = 0;
      // for each instance, copy data and fill sequence positions
      for (int instance = 0; instance < cpySize; instance++) {
        int64_t currInstanceId = seqs[instance];
        int64_t currInstanceLength =
            indexs[currInstanceId + 1] - indexs[currInstanceId];
        sparse_non_value_t* currInstanceData = data + indexs[currInstanceId];
        // write sequenceStartPositions
        allSequenceLength += currInstanceLength;
        currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
        // copy features
        for (int featCopier = 0; featCopier < currInstanceLength;
             featCopier++) {
          currPosOfArgumentId[featCopier] = currInstanceData[featCopier].col;
        }
        currPosOfArgumentId += currInstanceLength;
        // special: if current instance has NO feature in current slot
        if (currInstanceLength == 0) {
          allSequenceLength++;
          currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
          currPosOfArgumentId[0] = -1;
          currPosOfArgumentId++;
        }
        // done
      }  // end for every instance
    }    // end for slot

    // label
    IVector::resizeOrCreate(cpuArguments[slots_.size()].ids, cpySize,
                            /* useGpu= */ false);

    // fill labels
    int* label =
        (labelSeqs_.size() ? labelSeqs_.data() : labels_.data()) + startIndex;
    cpuArguments[slots_.size()].ids->copyFrom(label, cpySize);
    // label HAS sequence structure
    ICpuGpuVector::resizeOrCreate(
        cpuArguments[slots_.size()].sequenceStartPositions, cpySize + 1,
        /* useGpu= */ false);
    cpuArguments[slots_.size()].sequenceStartPositions->fillSequence(false);

    // info
    IVector::resizeOrCreate(cpuArguments[slots_.size() + 1].ids, cpySize,
                            /* useGpu= */ false);
    // info HAS sequence structure
    ICpuGpuVector::resizeOrCreate(
        cpuArguments[slots_.size() + 1].sequenceStartPositions, cpySize + 1,
        /* useGpu= */ false);
    cpuArguments[slots_.size() + 1].sequenceStartPositions->fillSequence(false);
    // fill infos
    int* info =
        (infoSeqs_.size() ? infoSeqs_.data() : infos_.data()) + startIndex;
    cpuArguments[slots_.size() + 1].ids->copyFrom(info, cpySize);
  }
  *batch = cpuBatch;
  batch->setSize(cpySize);
  return cpySize;
}

}  // namespace paddle
