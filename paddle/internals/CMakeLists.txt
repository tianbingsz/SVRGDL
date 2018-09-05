# Baidu Internals Libs
# 
# NOTE the current dir is parent dir, ${PROJ_ROOT}
option(WITH_PREDICT_SDK "Compile paddle with predict sdk (which is deprecated)." OFF)
option(WITH_METRIC "Compile paddle use metric learning" OFF)
option(WITH_FPGA "With FPGA or not" OFF)
set(ULLIB_PATH $ENV{ULLIB_PATH} CACHE PATH "")
if(WITH_FPGA)
    include_directories(${ULLIB_PATH}/include)
    add_definitions(-DPADDLE_USE_FPGA)
    include_directories(${PROJ_ROOT}/paddle/internals/fpga_output/include/dnn-lib)
    include_directories(${PROJ_ROOT}/paddle/internals/fpga_output/include/dnn-api)
endif()

set(INTERNAL_PROTO_PATH "${PROJ_ROOT}/paddle/internals/proto")

if(WITH_METRIC)
   add_definitions(-DPADDLE_WITH_METRIC_LEARNING)
   link_directories("${PROJ_ROOT}/paddle/internals/metric_lib")
   add_subdirectory(paddle/internals/metric_learning)
   add_subdirectory(paddle/internals/dserver)
endif()
add_subdirectory(paddle/internals/gserver)
add_subdirectory(paddle/internals/parameter)
add_subdirectory(paddle/internals/owlqn)
add_subdirectory(paddle/internals/trainer)
