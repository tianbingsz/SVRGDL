/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

// This class is not in paddle namespace for backward compability to metric
// learning code.

/**
 * @brief Light-weight inter Matrix
 */
class InterMatrix {
protected:
  real* data_;
  int height_, width_;
  bool trans_;

public:
  InterMatrix() : data_(NULL), height_(0), width_(0), trans_(false) {}
  InterMatrix(real* data, int height, int width, bool trans)
      : data_(data), height_(height), width_(width), trans_(trans) {}
  int getHeight() const { return height_; }
  int getWidth() const { return width_; }
  int getNumElements() const { return height_ * width_; }
  bool isTrans() const { return trans_; }
  real* getData() const { return data_; }
  int getLeadingDim() const { return trans_ ? height_ : width_; }
  void reshape(int height, int width) {
    CHECK_EQ(height_ * width_, height * width);
    height_ = height;
    width_ = width;
  }
};
