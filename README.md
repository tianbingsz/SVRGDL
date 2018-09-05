# PaddleSVRG
* Author: Tianbing Xu, Baidu Research, CA
* Code Built: Oct. 2016
## Motivation
For SGD, the high variance is an important problem to slow down the
Neureal Network training. Recently, stochastic variance reduction technique
was proposed in machine learning community and achieved improved empirical performance 
in Convex Optimization problems. However, we are not clear,
whether variance reduction is able to improve the non-convex problem empirically 
such as Deep Learning? We implemented variance reduction in Paddle,
made heavy code changes in different parts, including GradientMachine, Optimizer, 
Trainer, Parameter Server and so on. The preliminary results are encouraging.
For Multi-layer perceptons (MLP), the training is improved with sample efficiency 
and higher accuracy. The testing accuracy is improved marginally. Further work is to investigate 
the variance reduction impacts on large scale training of industrial models such as ResNet.

## Updated or Newly added Code based on Paddle
* paddle/gserver/gradientmachines/GradientMachine.h, paddle/gserver/gradientmachines/MultiGradientMachine.cpp

* paddle/math/BaseMatrix.cu, paddle/math/BaseMatrix.h

* paddle/parameter/FirstOrderOptimizer.h, paddle/parameter/ParameterOptimizer.cpp

* paddle/pserver/ParameterServer2.cpp, paddle/pserver/ParameterServer2.h

* paddle/trainer/CMakeLists.txt, paddle/trainer/RemoteParameterUpdaterVR.cpp, paddle/trainer/RemoteParameterUpdaterVR.h, 
paddle/trainer/Trainer.h, paddle/trainer/TrainerInternal.h, paddle/trainer/TrainerInternalVR.cpp,
paddle/trainer/TrainerInternalVR.h, paddle/trainer/TrainerMainVR.cpp, paddle/trainer/TrainerVR.cpp, paddle/trainer/TrainerVR.h

* paddle/utils/GlobalConstants.cpp, paddle/utils/GlobalConstants.h

* proto/ParameterService.proto.m4, proto/TrainerConfig.proto.m4

* python/paddle/trainer/config_parser.py


## REFERENCE
R. Johnson and T. Zhang, “Accelerating stochastic gradient descent using predictive variance reduction, NIPS 2013
