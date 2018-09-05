/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#include <paddle/utils/PythonUtil.h>

#include <gtest/gtest.h>

#include "paddle/trainer/Trainer.h"
#include "paddle/math/ImportanceSampler.h"

using namespace std;
using namespace paddle;

size_t checkData(real* data1, real* data2, size_t len) {
  size_t num = 0;
  for (size_t i = 0; i < len; i++) {
    if (data1[i] != data2[i]) {
      num++;
    }
  }
  return num;
}

void testImportanceSampler(size_t numInstances, size_t numSamples,
                           unsigned int randSeed,
                           bool keepWeightsAfterSampling) {
  unique_ptr<RandomNumberGenerator> randGen(
      new RandomNumberGeneratorMT19937_64(randSeed));
  ImportanceSamplerWithoutReplacement sampler(randGen,
                                              keepWeightsAfterSampling);
  double sumW = sampler.init(numInstances);
  LOG(INFO) << "\n\nnumInstances " << numInstances << ", numSamples "
            << numSamples << ", randSeed " << randSeed;
  LOG(INFO) << "sumW: " << sumW;
  vector<size_t> sampleIndices(numSamples);
  vector<double> sampleWeights(numSamples);
  size_t numSampled;
  double smpW = sampler.sampling(numSamples, numSampled, &sampleIndices[0],
                                 &sampleWeights[0]);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  string info;
  char buf[1024];
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;
  for (size_t i = 0; i < numSampled; i++) {
    sampleWeights[i] = i;
  }

  smpW = sampler.sampling(numSamples, numSampled, &sampleIndices[0],
                          &sampleWeights[0]);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  info = string("");
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;
  for (size_t i = 0; i < numSampled; i++) {
    sampleWeights[i] = i;
  }

  double newSumW =
      sampler.updateWeights(numSampled, &sampleIndices[0], &sampleWeights[0]);
  LOG(INFO) << "newSumW: " << newSumW;

  smpW = sampler.sampling(numSamples, numSampled, &sampleIndices[0],
                          &sampleWeights[0]);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  info = string("");
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;

  size_t num = sampler.getNumInstances();
  LOG(INFO) << "num: " << num;
  vector<double> instW(num);
  sumW = sampler.getWeightInstances(&instW[0]);
  info = string("");
  for (size_t i = 0; i < std::min(size_t(20), num); i++) {
    sprintf(buf, "%.6f", instW[i]);
    info += buf;
  }
  LOG(INFO) << info;
}

void testImportanceSamplerVec(size_t numInstances, size_t numSamples,
                              unsigned int randSeed,
                              bool keepWeightsAfterSampling) {
  unique_ptr<RandomNumberGenerator> randGen(
      new RandomNumberGeneratorMT19937_64(randSeed));
  ImportanceSamplerWithoutReplacement sampler(randGen,
                                              keepWeightsAfterSampling);
  double sumW = sampler.init(numInstances);
  LOG(INFO) << "\n\nnumInstances " << numInstances << ", numSamples "
            << numSamples << ", randSeed " << randSeed;
  LOG(INFO) << "sumW: " << sumW;
  vector<size_t> sampleIndices;
  vector<double> sampleWeights;
  size_t numSampled;
  double smpW =
      sampler.samplingVec(numSamples, numSampled, sampleIndices, sampleWeights);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  string info;
  char buf[1024];
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;
  for (size_t i = 0; i < numSampled; i++) {
    sampleWeights[i] = i;
  }

  smpW =
      sampler.samplingVec(numSamples, numSampled, sampleIndices, sampleWeights);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  info = string("");
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;
  for (size_t i = 0; i < numSampled; i++) {
    sampleWeights[i] = i;
  }

  double newSumW =
      sampler.updateWeightsVec(numSampled, sampleIndices, sampleWeights);
  LOG(INFO) << "newSumW: " << newSumW;

  smpW =
      sampler.samplingVec(numSamples, numSampled, sampleIndices, sampleWeights);
  LOG(INFO) << "smpW: " << smpW << ", numSampled: " << numSampled;
  info = string("");
  for (size_t i = 0; i < numSampled; i++) {
    sprintf(buf, "%ld:%.6f ", sampleIndices[i], sampleWeights[i]);
    info += buf;
  }
  LOG(INFO) << info;

  vector<double> instW;
  sumW = sampler.getWeightInstancesVec(instW);
  size_t num = instW.size();
  info = string("");
  for (size_t i = 0; i < std::min(size_t(20), num); i++) {
    sprintf(buf, "%.6f ", instW[i]);
    info += buf;
  }
  LOG(INFO) << info;
}

TEST(ImportanceSamplery, test) {
  testImportanceSampler(100, 10, 0, true);
  testImportanceSampler(100, 10, 0, false);
  testImportanceSamplerVec(100, 10, 0, false);
  testImportanceSampler(5, 10, 1, false);
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  // hl_start();
  // hl_init(FLAGS_gpu_id);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
