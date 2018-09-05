/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "paddle/utils/TypeDefs.h"

namespace paddle {

/**
 * A set of voice data files which have been marked
 */
class DataFileStruct {
public:
  char featFileName[1024];
  char labelFileName[1024];
  char descFileName[1024];

public:
  DataFileStruct() {}

  /**
   * inFeat: the filename of feature data
   * inLabel: the filename of label data
   * inDesc: the filename of destination data
   *   Describe how many frames feature in a voice file
   */
  DataFileStruct(const char* inFeat, const char* inLabel, const char* inDesc) {
    strcpy(featFileName,  // NOLINT
           inFeat);       // NOLINT TODO(yuyang18): use snprintf instead.
    strcpy(labelFileName, inLabel);  // NOLINT
    strcpy(descFileName, inDesc);    // NOLINT
  }

  DataFileStruct(const DataFileStruct& inC) { *this = inC; }

  ~DataFileStruct() {}

  DataFileStruct& operator=(const DataFileStruct& inC) {
    strcpy(featFileName, inC.featFileName);    // NOLINT
    strcpy(labelFileName, inC.labelFileName);  // NOLINT
    strcpy(descFileName, inC.descFileName);    // NOLINT
    return (*this);
  }
};
typedef std::vector<DataFileStruct> FileNameVector;

/**
 *
 */
class CepstrumExtract {
protected:
  // feature buffer ( all features are stored in one huge vector)
  real* featDataBuffer_;
  // expected size of pFeatContent
  // expected Large-frame number*(2*NFRAME+1)*oneFrmDim
  int64_t dataBufferSize_;
  // feature element number in feature buffer
  int64_t featElementNum_;

  // each samll frame's starting position
  // (samll frame size = 39 or 42 for MFCC)
  std::vector<int64_t> featMarkPos_;
  // original label of training data
  std::vector<int> featLabelVec_;
  // utterance block start position in feature buffer
  // ( one block: 10000 )
  std::vector<int64_t> uttBlkStartPos_;

  // stochastic position
  // index of label of each frames which will be randomized
  std::vector<int64_t> stochasticPos_;
  // stochastic Pos's iterator
  std::vector<int64_t>::iterator iter_;

  // part of data used in training
  // if not -1, part of train data is used in training
  int partTrDataNum_;

  // max neighbour frame numbers
  // e.g. 5 means left 5 context and right 5 context
  int contextFrmLen_;
  // feature dimension of one frame
  int oneFrmDim_;

  FileNameVector fileVector_;
  int fileStartPos_;
  int fileLoadNum_;

protected:
  /**
   * read all train data
   */
  void readFeat(std::vector<DataFileStruct>& fileNameVector, int startPos,
                int loadSize);
  void readMark(std::vector<DataFileStruct>& fileNameVector, int startPos,
                int loadSize);
  void ReadDescFile(std::vector<DataFileStruct>& fileNameVector, int startPos,
                    int loadSize);

  /**
   *read one file info (cepstrum, mark, desc)
   */
  void readFeat(const char* filename);
  uint64_t readMark(const char* filename, uint64_t offset);
  void ReadEachDescFile(const char* partialFile, int fileNo);

public:
  // Constructor and destructor
  CepstrumExtract(int inFrameSize, int inContextFramLen = 5,
                  int inTrainNum = -1);

  ~CepstrumExtract();

  /**
   * Set the train fileList
   */
  void setDataFileList(const char* dataFileListName, int fileLoadNum);

  /**
   * read fileLoadNum_ files
   * return true if some files are read.
   * return false if there no more files.
   */
  bool readData();

  /**
   *random the data buffer
   */
  void randomize();

  /**
   *reset the data to init status
   */
  void reset();

  /**
   *load data to buffer
   */
  int write2Buf(real* featBuf, int* labelBuf);

  /**
   *to judge if buffer is finished
   */
  bool eof();

private:
  /* Removing the spaces in the string */
  void removeWhiteSpaceAndComment(char* str);

  /* Bytes swap */
  inline void swap32(char* c) {
    char t0 = c[0];
    char t1 = c[1];
    c[0] = c[3];
    c[1] = c[2];
    c[2] = t1;
    c[3] = t0;
  }
};

typedef std::shared_ptr<CepstrumExtract> CepstrumExtractPtr;
}  // namespace paddle
