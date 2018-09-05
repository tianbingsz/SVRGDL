/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include <algorithm>
#include <stdlib.h>
#include <byteswap.h>

#include "paddle/utils/Logging.h"

#include "CepstrumExtract.h"

namespace paddle {

CepstrumExtract::CepstrumExtract(int innOneFrmDim, int inContextFrmLen,
                                 int inPartTrNum) {
  /* init iter */
  iter_ = stochasticPos_.begin();

  /* buffer */
  featDataBuffer_ = NULL;
  dataBufferSize_ = 1000000;

  /* allocate memory buffer */
  featElementNum_ = 0;
  featDataBuffer_ = (real*)malloc(dataBufferSize_ * sizeof(real));
  CHECK(featDataBuffer_) << "CepstrumExtract init: can't allocate enought mem";

  oneFrmDim_ = innOneFrmDim;

  /* set context frame len */
  /* 5 means that left-context is
   * 5 frame and right-conttext is also 5 frame. */
  contextFrmLen_ = inContextFrmLen;

  /* set debug test data number */
  /* means that the part of training data is used for training. */
  /* -1 means all data are used */
  partTrDataNum_ = inPartTrNum;
}

CepstrumExtract::~CepstrumExtract() {
  if (featDataBuffer_ != NULL) {
    free(featDataBuffer_);
    featDataBuffer_ = NULL;
  }
}

bool CepstrumExtract::eof() {
  /* for ML value and gradient calculation */
  if (partTrDataNum_ > 0) {  // Only part of the data is used for training
    return (iter_ == stochasticPos_.end()) ||
           (iter_ - stochasticPos_.begin() >= partTrDataNum_);
  } else {  // whole data is used for training
    return iter_ == stochasticPos_.end();
  }
}

void CepstrumExtract::reset() {
  /* One loop is defined that the gradients of W are kept to be fixed */
  fileStartPos_ = 0;
  iter_ = stochasticPos_.end();
}

int CepstrumExtract::write2Buf(real* featBuf, int* labelBuf) {
  int64_t curPos;
  real* pbuf = featBuf;
  if (iter_ != stochasticPos_.end()) {
    int64_t t = featMarkPos_[*iter_];
    iter_++;
    curPos = t;
    real* p = featDataBuffer_ + curPos - contextFrmLen_ * oneFrmDim_;
    for (int j = 0; j < (2 * contextFrmLen_ + 1) * oneFrmDim_; ++j, ++pbuf)
      *pbuf = (real)p[j];
    *labelBuf = featLabelVec_[curPos / oneFrmDim_];
    /*
     * "-1" is added to the head and tail of each utterance for context
     * extension.
     * If context 5 is considered, then there will be 11 frames for each
     * feature.
     * We add 5 frames to the head and 5 frame to the tail. These 10 frames's
     * lable
     * will be set to be -1.
     */
    CHECK(*labelBuf != -1) << "CepstrumExtract::write2Buf\t feature index "
                           << "error and label could't be -1: curPos = "
                           << curPos;
  } else {
    return false;

    // load data again
    // set iter
    // write2Buf()
  }
  return true;
}

void CepstrumExtract::setDataFileList(const char* dataFileListName,
                                      int fileLoadNum) {
  std::ifstream dataFile(dataFileListName);
  CHECK(dataFile) << "Fail to open " << dataFileListName;
  char* str = (char*)malloc(1024 * sizeof(char));
  str[0] = 0;
  while (!dataFile.eof()) {
    DataFileStruct* dataFileStruct = new class DataFileStruct;

    dataFile.getline(str, 1024);
    removeWhiteSpaceAndComment(str);
    if ('\0' == str[0]) {
      continue;
    }
    sscanf(str, "%s", dataFileStruct->featFileName);

    dataFile.getline(str, 1024);
    removeWhiteSpaceAndComment(str);
    if ('\0' == str[0]) {
      continue;
    }
    sscanf(str, "%s", dataFileStruct->labelFileName);

    dataFile.getline(str, 1024);
    removeWhiteSpaceAndComment(str);
    if ('\0' == str[0]) {
      continue;
    }
    sscanf(str, "%s", dataFileStruct->descFileName);

    fileVector_.push_back(*dataFileStruct);
    delete dataFileStruct;
  }

  if (fileLoadNum <= 0)
    fileLoadNum_ = fileVector_.size();
  else
    fileLoadNum_ = fileLoadNum;
  fileStartPos_ = 0;

  dataFile.close();
  free(str);
  str = NULL;
}

bool CepstrumExtract::readData() {
  featElementNum_ = 0;
  uttBlkStartPos_.clear();
  LOG(INFO) << "Loading data";
  if ((uint32_t)fileStartPos_ < fileVector_.size()) {
    readFeat(fileVector_, fileStartPos_, fileLoadNum_);
    readMark(fileVector_, fileStartPos_, fileLoadNum_);
    ReadDescFile(fileVector_, fileStartPos_, fileLoadNum_);
    fileStartPos_ += fileLoadNum_;
    VLOG(1) << "=====================================" << std::endl
            << "featElementNum " << featElementNum_ << std::endl
            << "featLabelVec: " << featLabelVec_.size() << std::endl
            << "featMarkPos: " << featMarkPos_.size() << std::endl
            << "=====================================";
    return true;
  } else {
    return false;
  }
}

size_t getFileSize(const std::string& filename) {
  std::ifstream f(filename, std::ios::binary);
  CHECK(f) << "failed to open " << filename;
  f.seekg(0, std::ios::end);
  return f.tellg();
}

void CepstrumExtract::readFeat(std::vector<DataFileStruct>& fileNameVector,
                               int startPos, int loadSize) {
  /* read block utterances of loadsize
   * blocks (one block = 10000 utternace) */
  int fileVecSz = fileNameVector.size();

  size_t totalSize = 0;
  for (int i = 0; i < loadSize && (i + startPos) < fileVecSz; i++) {
    totalSize += getFileSize(fileNameVector[i + startPos].featFileName);
  }
  free(featDataBuffer_);
  featDataBuffer_ = (real*)malloc(totalSize);
  CHECK(featDataBuffer_) << "Cannot allocate memory " << totalSize << " bytes";
  dataBufferSize_ = totalSize / sizeof(real);

  for (int i = 0; i < loadSize && (i + startPos) < fileVecSz; i++) {
    DataFileStruct& fileName = fileNameVector[i + startPos];
    readFeat(fileName.featFileName);
  }
  /* shrink memory */
  real* temp = (real*)realloc(featDataBuffer_, featElementNum_ * sizeof(real));
  CHECK(temp != NULL) << "function readFeat: can't realloc mem "
                      << featElementNum_ * sizeof(real) << " bytes";
  featDataBuffer_ = temp;
  uttBlkStartPos_.push_back(featElementNum_);

  /* swap bytes ( feature is in float format and big-end ) */
  for (int64_t ii = 0; ii < featElementNum_; ++ii) {
    // featDataBuffer_[ii] = swap_32(featDataBuffer_[ii]);
    swap32((char*)&featDataBuffer_[ii]);
  }
}

void CepstrumExtract::readFeat(const char* filename) {
  std::ifstream inFeatFile(filename, std::ios::binary);
  CHECK(inFeatFile) << "failed to open " << filename;
  uttBlkStartPos_.push_back(featElementNum_);
  inFeatFile.seekg(0, std::ios::end);
  int64_t len = inFeatFile.tellg();
  inFeatFile.seekg(0, std::ios::beg);
  if (featElementNum_ + len / (int)sizeof(real) > dataBufferSize_) {
    VLOG(1) << "CepstrumExtract::readFeat\t want allocate memeory size : "
            << ((featElementNum_ * sizeof(real) + len) * 1.5 / (1024.0 * 1024))
            << "M";
    dataBufferSize_ = (featElementNum_ + len / sizeof(real)) * 1.5;
    real* temp = (real*)malloc(dataBufferSize_ * sizeof(real));
    CHECK(temp != NULL) << "Function readFeat can't allocate enough memory";
    memcpy(temp, featDataBuffer_, featElementNum_ * sizeof(real));
    free(featDataBuffer_);
    featDataBuffer_ = temp;
  }
  /* read data from file to &pFeatDataBuffer[featElementNum_] */
  inFeatFile.read((char*)(featDataBuffer_ + featElementNum_), len);

  featElementNum_ += len / sizeof(real);
  inFeatFile.close();
  VLOG_EVERY_N(1, 1000) << "Read PLP feature with nfile index = "
                        << uttBlkStartPos_.size();
}

void CepstrumExtract::readMark(std::vector<DataFileStruct>& fileNameVector,
                               int startPos, int loadSize) {
  featLabelVec_.resize(featElementNum_ / oneFrmDim_);
  uint64_t offset = 0;
  int fileVecSz = fileNameVector.size();
  for (int i = 0; i < loadSize && (i + startPos) < fileVecSz; i++) {
    DataFileStruct& fileName = fileNameVector[i + startPos];
    offset += readMark(fileName.labelFileName, offset);
  }
  CHECK_EQ(featLabelVec_.size(), offset) << "Wrong: feat/label file mismatch";
}

uint64_t CepstrumExtract::readMark(const char* filename, uint64_t offset) {
  std::ifstream inFile(filename, std::ios::binary);
  if (inFile.fail()) {
    perror(filename);
    return -1;
  }

  inFile.seekg(0, std::ios::end);
  uint64_t len = inFile.tellg();
  inFile.seekg(0, std::ios::beg);
  int curItem = -1;
  uint64_t cnt = 0;
  uint64_t size = len / (int)sizeof(curItem);
  CHECK_LE(offset + size, featLabelVec_.size())
      << "Wrong: feat/label file mismatch";

  for (; cnt < len / (int)sizeof(curItem); ++cnt) {
    inFile.read((char*)&curItem, sizeof(curItem));
    featLabelVec_[cnt + offset] = curItem;
  }
  inFile.close();
  return cnt;
}

void CepstrumExtract::ReadDescFile(std::vector<DataFileStruct>& fileNameVector,
                                   int startPos, int loadSize) {
  featMarkPos_.clear();
  int fileVecSz = fileNameVector.size();
  for (int i = 0; i < loadSize && (i + startPos) < fileVecSz; i++) {
    DataFileStruct& fileName = fileNameVector[i + startPos];
    ReadEachDescFile(fileName.descFileName, i);
  }
}

void CepstrumExtract::ReadEachDescFile(const char* descFileName, int fileNo) {
  char str[4096], fileName[4096];
  std::ifstream descFileStream(descFileName);
  int nBlock = -1, stPosition = -1, nFrameLen = -1;
  if (descFileStream.fail()) {
    perror(descFileName);
    exit(1);
  }

  /* To calculate featMarkPos_ */
  while (descFileStream.getline(str, 4096)) {
    /* NOTE data format:
       *File formate: fileName is the cepstrum binary file of one utterance
       *nBlock      : is the large-block train data id (10000 cepstrum files
      *form one block of train data)
       *offset      : starting frame id of the utterance  in the large-block
      *train data
       *nFrameLen   : cepstrum length of the utterance */
    sscanf(str, "%s %d%d%d", fileName, &nBlock, &stPosition, &nFrameLen);

    int64_t fileStartPos = uttBlkStartPos_[fileNo] + stPosition * oneFrmDim_;
    for (int i = 0; i < nFrameLen; ++i) {
      int64_t curPos = fileStartPos + i * oneFrmDim_;
      CHECK(featLabelVec_[curPos / oneFrmDim_] != -1)
          << "in " << nBlock << " join feature file: wrong class ID(-1) at "
          << curPos / oneFrmDim_ << " frames";
      featMarkPos_.push_back(curPos);
    }
  }  // while
}

void CepstrumExtract::randomize() {
  stochasticPos_.clear();
  stochasticPos_.resize(featMarkPos_.size());
  for (int64_t i = 0; i < (int64_t)featMarkPos_.size(); ++i)
    stochasticPos_[i] = i;
  std::random_shuffle(featMarkPos_.begin(), featMarkPos_.end());
  iter_ = stochasticPos_.begin();
}

void CepstrumExtract::removeWhiteSpaceAndComment(char* str) {
  char temp[2048];
  char* p = str;
  int i = 0;

  // remove whitespace
  while (*p) {
    if (!isspace(*p)) temp[i++] = *p;
    ++p;
  }
  temp[i] = '\0';

  // remove the comment after '#'
  if ((p = strchr(temp, '#')) != NULL) *p = '\0';
  strcpy(str, temp);  // NOLINT TODO(yuyang18): use snprintf instead.
}
}  // namespace paddle
