/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */

#include "TrainerOWLQN.h"

#include <deque>

namespace paddle {

void OWLQN::waitGradient() {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_COPY, internalGrad_, newgrad_);
  client_.doOperation(ops, true, false);
}

void OWLQN::sendBackNewValue() {
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_COPY, newx_, internalX_);
  client_.doOperation(ops, false, true);
}

real OWLQN::getCost() {
  real ret = 0.0;
  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_COST, x_, newgrad_, l1weight_, l2weight_)(&ret);
  client_.doOperation(ops, false, false);
  return ret;
}

void OWLQN::train(bool& accepted) {
  // objective value controller
  real c1 = config_.c1();
  real backoff = config_.backoff();
  int maxBackoff = config_.max_backoff();
  real step = 1.0;
  real oldobj = 0;
  real newobj = 0;
  int alwaysBackoffCount = 0;
  real origDirDeriv = 0;
  bool isiter0 = true;
  bool wolfeok = true;
  bool needNewDir = true;

  while (true) {
    if (passCount_ == expectedPassCount_) {
      break;
    }
    passCount_++;
    // copy internalGrad to newgrad_
    waitGradient();
    if (isiter0) {
      // copy internalX to x_ and newx_, internalGrad to grad_
      client_.vectorCopy(internalX_, x_);
      client_.vectorCopy(internalX_, newx_);
      client_.vectorCopy(internalGrad_, grad_);
    }
    // current objective value
    newobj = getCost();
    LOG(INFO) << "objective_value=" << newobj;
    if (isiter0) {
      // do not consider wolfe condition (always accept)
      oldobj = newobj;
      needNewDir = true;
      accepted = true;
    } else {
      // wolfe condition
      if (alwaysBackoffCount == maxBackoff ||
          newobj <= oldobj + c1 * origDirDeriv * step) {
        wolfeok = true;
        oldobj = newobj;
        alwaysBackoffCount = 0;
      } else {
        wolfeok = false;
        alwaysBackoffCount++;
      }
      LOG(INFO) << "wolfe condition test result: " << wolfeok;
      accepted = wolfeok;
      if (wolfeok) {
        needNewDir = true;
        shift();
      } else {
        // not accepted, try smaller step
        step *= backoff;
        needNewDir = false;
      }
    }
    if (needNewDir) {
      step = 1.0;
      updateDir();
      // check dir_
      origDirDeriv = dirDeriv();
      if (origDirDeriv >= 0) {
        LOG(FATAL) << "check your gradient!";
        exit(1);
      }
    }
    // GetNextPoint
    getNextPoint(isiter0, &step);
    // copy newx_ to internalX
    sendBackNewValue();
    isiter0 = false;
  }
}

void OWLQN::init() {
  steepestDescDir_ = createVector();
  dir_ = createVector();
  x_ = createVector();
  newx_ = createVector();
  grad_ = createVector();
  newgrad_ = createVector();
  l1weight_ = config_.l1weight();
  l2weight_ = config_.l2weight();
  l2weightBackup_ = l2weight_;
  internalIter_ = 0;
  passCount_ = 0;
  l2weightZeroIter_ = config_.l2weight_zero_iter();
  owlqnSteps_ = config_.owlqn_steps();
  alphas_.resize(owlqnSteps_);
  // internal vector of x_ and grad_
  internalX_ = client_.getPServerParameterValue();
  internalGrad_ = client_.getPServerParameterGradient();
}

void OWLQN::deinit() {
  client_.releaseVector(steepestDescDir_);
  client_.releaseVector(dir_);
  client_.releaseVector(x_);
  client_.releaseVector(newx_);
  client_.releaseVector(grad_);
  client_.releaseVector(newgrad_);
  for (size_t i = 0; i < slist_.size(); i++) {
    client_.releaseVector(slist_[i]);
    client_.releaseVector(ylist_[i]);
  }
}

void OWLQN::shift() {
  internalIter_++;
  LOG(INFO) << "new internalIter_=" << internalIter_;
  if (l2weightZeroIter_ > 0) {
    if (internalIter_ > l2weightZeroIter_) {
      l2weight_ = 0;
    } else {
      l2weight_ =
          l2weightBackup_ * (1.0 - 1.0 * internalIter_ / l2weightZeroIter_);
    }
    LOG(INFO) << "new l2weight_=" << l2weight_;
  }
  int listsize = (int)slist_.size();
  PServerVector news, newy;
  if (listsize < owlqnSteps_) {
    news = createVector();
    newy = createVector();
  } else {
    news = slist_.front();
    slist_.pop_front();
    newy = ylist_.front();
    ylist_.pop_front();
    roList_.pop_front();
  }
  // calc news, newy, new ro
  client_.vectorAddMultInto(news, newx_, x_, -1);
  client_.vectorAddMultInto(newy, newgrad_, grad_, -1);
  real ro = client_.vectorDotProduct(news, newy);
  // push into vectors
  slist_.push_back(news);
  ylist_.push_back(newy);
  roList_.push_back(ro);
  // swap x_ and newx_. swap grad_ and newgrad_.
  std::swap(x_, newx_);
  std::swap(grad_, newgrad_);
}

void OWLQN::updateDir() {
  makeSteepestDescDir();
  mapDirByInverseHessian();
  fixDirSigns();
}

void OWLQN::makeSteepestDescDir() {
  if (l1weight_ == 0) {
    client_.vectorScaleInto(dir_, grad_, -1);
  } else {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_MAKE_STEEPEST_DESC_DIR, dir_, grad_, x_,
                     l1weight_);
    client_.doOperation(ops, false, false);
  }
  // copy dir_ to steepestDescDir_
  client_.vectorCopy(dir_, steepestDescDir_);
}

void OWLQN::mapDirByInverseHessian() {
  int count = slist_.size();
  if (count != 0) {
    for (int i = count - 1; i >= 0; i--) {
      real result = client_.vectorDotProduct(slist_[i], dir_);
      alphas_[i] = -result / roList_[i];
      client_.vectorAddMult(dir_, ylist_[i], alphas_[i]);
    }
    real yDotY = client_.vectorDotProduct(ylist_[count - 1], ylist_[count - 1]);
    real scalar = roList_[count - 1] / yDotY;
    client_.vectorScale(dir_, scalar);
    for (int i = 0; i < count; i++) {
      real beta = client_.vectorDotProduct(ylist_[i], dir_) / roList_[i];
      client_.vectorAddMult(dir_, slist_[i], -alphas_[i] - beta);
    }
  }
}

void OWLQN::fixDirSigns() {
  if (l1weight_ > 0) {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_FIX_DIR_SIGNS, dir_, steepestDescDir_);
    client_.doOperation(ops, false, false);
  }
}

void OWLQN::fixOmegaSigns() {
  if (l1weight_ > 0) {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_FIX_OMEGA_SIGNS, x_, newx_);
    client_.doOperation(ops, false, false);
  }
}

real OWLQN::dirDeriv() {
  real dirDeriv = 0;
  if (l1weight_ == 0) {
    dirDeriv = client_.vectorDotProduct(dir_, grad_);
  } else {
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_DIR_DERIV, dir_, grad_, x_,
                     l1weight_)(&dirDeriv);
    client_.doOperation(ops, false, false);
  }
  LOG(INFO) << "dirDeriv=" << dirDeriv;
  return dirDeriv;
}

void OWLQN::getNextPoint(bool isiter0, real* step) {
  if (isiter0 == true) {
    real normDir = sqrt(client_.vectorDotProduct(dir_, dir_));
    *step = 1.0 / normDir;
  }
  LOG(INFO) << "step=" << *step;
  client_.vectorAddMultInto(newx_, x_, dir_, *step);
  fixOmegaSigns();
}

}  // namespace paddle
