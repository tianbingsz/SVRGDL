/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#pragma once

#include "paddle/pserver/ParameterClient2.h"

#include <deque>

namespace paddle {

class OWLQN {
public:
  OWLQN(ParameterClient2& client, const OptimizationConfig& config,
        int expectedPassCount)
      : client_(client),
        config_(config),
        expectedPassCount_(expectedPassCount) {}
  void init();
  void deinit();
  void shift();
  void updateDir();
  void makeSteepestDescDir();
  void mapDirByInverseHessian();
  void fixDirSigns();
  void fixOmegaSigns();
  real dirDeriv();
  void getNextPoint(bool isiter0, real* step);
  void train(bool& accepted);
  void waitGradient();
  void sendBackNewValue();
  real getCost();
  PServerVector createVector() {
    PServerVector ret = client_.createVector();
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_RESET, ret, (real)0);
    client_.doOperation(ops, false, false);
    return ret;
  }
  PServerVector& newgrad() { return newgrad_; }
  PServerVector& x() { return x_; }
  PServerVector& grad() { return grad_; }
  PServerVector& newx() { return newx_; }

private:
  std::deque<real> roList_;
  std::vector<real> alphas_;
  PServerVector steepestDescDir_, dir_, x_, newx_, grad_, newgrad_;
  PServerVector internalX_, internalGrad_;
  ParameterClient2& client_;
  OptimizationConfig config_;
  int owlqnSteps_;
  std::deque<PServerVector> slist_, ylist_;
  real l1weight_;
  real l2weight_;
  real l2weightBackup_;
  int internalIter_;  // accepted pass
  int l2weightZeroIter_;
  int expectedPassCount_;
  int passCount_;  // total pass
};

}  // namespace paddle
