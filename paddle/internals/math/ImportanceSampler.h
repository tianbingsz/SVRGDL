/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve. */


#pragma once

#include <vector>
#include <memory>
#include <random>

namespace paddle {

class RandomNumberGenerator {
public:
  RandomNumberGenerator() {}

  virtual ~RandomNumberGenerator() {}

  // generate a number between 0 and 1, uniformlly
  virtual double rand() = 0;
};

class RandomNumberGeneratorMT19937_64 : public RandomNumberGenerator {
public:
  RandomNumberGeneratorMT19937_64(unsigned int seed);

  virtual ~RandomNumberGeneratorMT19937_64() {}

  virtual double rand();

protected:
  std::mt19937_64 randGen_;
  std::uniform_real_distribution<double> dist_;
};

//
// An importance sampler selects a small portion of training samples based on
// their given weights, which is usually better than conventional uniformal
// sampling.
//

class ImportanceSampler {
public:
  // constructor: a random number genrator is provided
  ImportanceSampler(std::unique_ptr<RandomNumberGenerator> &randGen)
      : randGen_(std::move(randGen)) {}

  virtual ~ImportanceSampler() {}

  // initialization of the sampler: the number of instances and their weights
  //(optional and 1 by default) are given. The weight sum is returned.
  virtual double init(size_t numInstances, double *initWeights = NULL) = 0;

  // sampling based on current instance weights. numSampled, sampleIndices and
  // sampleWeights are the number of sampled instances, their indices and their
  // adjusted weights, whose sum is returned.
  virtual double sampling(size_t numSamples, size_t &numSampled,
                          size_t *sampleIndices, double *sampleWeights) = 0;
  double samplingVec(size_t numSamples, size_t &numSampled,
                     std::vector<size_t> &sampleIndices,
                     std::vector<double> &sampleWeights);

  // update weights of selected instances, whose number, indices and weights
  // are given by numSamples, sampleIndices and sampleWeights
  virtual double updateWeights(size_t numSamples, size_t *sampleIndices,
                               double *sampleWeights) = 0;
  double updateWeightsVec(size_t numSamples, std::vector<size_t> &sampleIndices,
                          std::vector<double> &sampleWeights);

  // get the number of instances
  virtual size_t getNumInstances() = 0;

  // get weights of instances: "weights" is allocated externally whose size
  // is at least getNumInstances(); the sum of weights is returned.
  virtual double getWeightInstances(double *weights) = 0;

  // get weights of instances: the vector "weights" will be resized according
  // to getNumInstances().
  double getWeightInstancesVec(std::vector<double> &weights);

protected:
  std::unique_ptr<RandomNumberGenerator> randGen_;
};

//
//  ImportanceSamplerWithoutReplacement is a sampler sampling weighted instances
//  without replacement. It is more efficient than conventional sampling with
//  replacement as an instance is sampled only once at most.
//

class ImportanceSamplerWithoutReplacement : public ImportanceSampler {
public:
  // Set keepWeightsAfterSampling to be true if you want to keep instance
  // weights unchanged after sampling; otherwise, weights of sampled instances
  // are set to be zero.
  ImportanceSamplerWithoutReplacement(
      std::unique_ptr<RandomNumberGenerator> &randGen,
      bool keepWeightsAfterSampling = false)
      : ImportanceSampler(randGen),
        keepWeightsAfterSampling_(keepWeightsAfterSampling) {}

  virtual double init(size_t numInstances, double *initWeights = NULL);
  virtual double sampling(size_t numSamples, size_t &numSampled,
                          size_t *sampleIndices, double *sampleWeights);
  virtual double updateWeights(size_t numSamples, size_t *sampleIndices,
                               double *sampleWeights);
  virtual size_t getNumInstances();
  virtual double getWeightInstances(double *weights);

protected:
  size_t binarySearch(const double rn) const;
  void updateOneLeaf(size_t j, double w);

protected:
  std::vector<double> wBDT_;       // memory for binary tree
  size_t numInstances_;            // the number of all instances
  size_t treeLevel_;               // the level of the entire binary tree
  bool keepWeightsAfterSampling_;  // whether keep weights of sampled instances
                                   // unchanged after sampling
};
}
