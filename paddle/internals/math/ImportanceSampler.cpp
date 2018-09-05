/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include "ImportanceSampler.h"
#include <memory.h>
#include <stdlib.h>
#include "paddle/utils/Logging.h"

using namespace paddle;
using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////

RandomNumberGeneratorMT19937_64::RandomNumberGeneratorMT19937_64(
    unsigned int seed) {
  randGen_.seed(seed);
}

double RandomNumberGeneratorMT19937_64::rand() { return dist_(randGen_); }

////////////////////////////////////////////////////////////////////////////////////////////////////

double ImportanceSampler::samplingVec(size_t numSamples, size_t &numSampled,
                                      std::vector<size_t> &sampleIndices,
                                      std::vector<double> &sampleWeights) {
  sampleIndices.resize(numSamples);
  sampleWeights.resize(numSamples);
  return sampling(numSamples, numSampled, &sampleIndices[0], &sampleWeights[0]);
}

double ImportanceSampler::updateWeightsVec(size_t numSamples,
                                           std::vector<size_t> &sampleIndices,
                                           std::vector<double> &sampleWeights) {
  CHECK(numSamples <= sampleIndices.size());
  CHECK(numSamples <= sampleWeights.size());
  return updateWeights(numSamples, &sampleIndices[0], &sampleWeights[0]);
}

double ImportanceSampler::getWeightInstancesVec(std::vector<double> &weights) {
  weights.resize(getNumInstances());
  return getWeightInstances(&weights[0]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t getBdtOffset(size_t level) { return (1Ull << (level)) - 1; }

double ImportanceSamplerWithoutReplacement::init(size_t numInstances,
                                                 double *initWeights) {
  CHECK(numInstances > 0) << "numInstances " << numInstances;

  numInstances_ = numInstances;
  treeLevel_ = 0;
  while ((1Ull << treeLevel_) < numInstances) treeLevel_++;
  treeLevel_++;

  size_t bdtSz = (1Ull << treeLevel_) - 1;

  if (wBDT_.size() < bdtSz) wBDT_.resize(bdtSz);

  double *bdt = &wBDT_[0];
  double *p = bdt + getBdtOffset(treeLevel_ - 1);
  if (initWeights) {
    // copy weights to leaves
    std::copy(initWeights, initWeights + numInstances, p);
  } else {
    // set weights to be 1 by default
    std::fill(p, p + numInstances, 1.0);
  }
  // set other leaves to be 0
  p += numInstances;
  std::fill(p, bdt + getBdtOffset(treeLevel_), 0.0);

  // calculate weights of parent nodes
  for (int lvl = treeLevel_ - 2; lvl >= 0; lvl--) {
    size_t begin = (1Ull << lvl) - 1;
    size_t end = begin + (1Ull << lvl);
    for (size_t i = begin; i < end; i++)
      bdt[i] = bdt[(i << 1) + 1] + bdt[(i << 1) + 2];
  }

  return bdt[0];
}

size_t ImportanceSamplerWithoutReplacement::binarySearch(
    const double rn) const {
  const double *bdt = &wBDT_[0];
  size_t j = 0;
  double val = rn;
  for (size_t lvl = 0; lvl != treeLevel_ - 1; lvl++) {
    size_t left = (j << 1) + 1;
    size_t right = (j << 1) + 2;
    CHECK_NE(bdt[j], 0) << "Error in binarySearch " << j << ", " << bdt[0];
    if (bdt[left] != 0 && val <= bdt[left]) {
      // left child node
      j = left;
    } else {
      val -= bdt[left];
      j = right;
    }
  }
  return j;
}

void ImportanceSamplerWithoutReplacement::updateOneLeaf(size_t j, double w) {
  double *bdt = &wBDT_[0];
  bdt[j] = w;
  for (int lvl = treeLevel_ - 2; lvl >= 0; lvl--) {
    j = (j - 1) >> 1;
    bdt[j] = bdt[(j << 1) + 1] + bdt[(j << 1) + 2];
  }
}

double ImportanceSamplerWithoutReplacement::sampling(size_t numSamples,
                                                     size_t &numSampled,
                                                     size_t *sampleIndices,
                                                     double *sampleWeights) {
  size_t i;
  double *bdt = &wBDT_[0];
  if (numSamples >= numInstances_) {
    // not enough instances for sampling
    numSampled = numInstances_;
    double *p = bdt + getBdtOffset(treeLevel_ - 1);
    for (i = 0; i < numInstances_; i++) {
      sampleIndices[i] = i;
      sampleWeights[i] = p[i];
    }
    return bdt[0];
  }

  numSampled = numSamples;
  double Ws = 0;       // weights sum of selected instances
  double Wr = bdt[0];  // weights sum of remained instances
  // seqW is for the weight sequence generated during sampling
  // sampleOriW is for storing weights of sampled instances.
  vector<double> seqW(numSamples), sampleOriW(numSamples);

  // sample instances sequentially
  for (i = 0; i < numSamples; i++) {
    if (i != 0) {
      // expection of number of times drawn from already sampled instances
      seqW[i - 1] = 1.0 / Wr;
    }
    // sample from the remained ones
    double rn = randGen_->rand() * Wr;
    // binary search in bdt
    size_t j = binarySearch(rn);
    CHECK_NE(bdt[j], 0) << "Error in binarySearch " << j << ", " << bdt[0];
    size_t id = j - getBdtOffset(treeLevel_ - 1);
    sampleIndices[i] = id;
    Ws += bdt[j];
    sampleOriW[i] = bdt[j];
    // update the leaf node, set it to be 0
    updateOneLeaf(j, 0);
    Wr = bdt[0];
  }

  CHECK_NE(Wr, 0) << "Error in binarySearch: final Wr == 0";
  seqW[i - 1] = 1.0 / Wr;

  double accSeqW = 0, sumW = 0;
  i = numSamples;
  while (i != 0) {
    i--;
    accSeqW += seqW[i];
    sumW += sampleWeights[i] = 1.0 + accSeqW * sampleOriW[i];
  }

  if (keepWeightsAfterSampling_) {
    updateWeights(numSamples, &sampleIndices[0], &sampleOriW[0]);
  }

  return sumW;
}

size_t ImportanceSamplerWithoutReplacement::getNumInstances() {
  return numInstances_;
}

double ImportanceSamplerWithoutReplacement::getWeightInstances(
    double *weights) {
  memcpy(weights, &wBDT_[0] + getBdtOffset(treeLevel_ - 1),
         sizeof(double) * numInstances_);
  return wBDT_[0];
}

// update weights of selected instances
double ImportanceSamplerWithoutReplacement::updateWeights(
    size_t numSamples, size_t *sampleIndices, double *sampleWeights) {
  size_t offset = getBdtOffset(treeLevel_ - 1);
  for (size_t i = 0; i < numSamples; i++)
    updateOneLeaf(sampleIndices[i] + offset, sampleWeights[i]);

  return wBDT_[0];
}
