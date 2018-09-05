# Gserver package contains:
#   * Layers
#   * Activations
#   * DataProviders
#   * Evaluators
#   * GradientMachines(NeuralNetwork)
file(GLOB_RECURSE INTERNAL_GSERVER_HEADER RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE INTERNAL_GSERVER_SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")

list(REMOVE_ITEM INTERNAL_GSERVER_SOURCES
        extension/FcrCvmDataProvider.cpp)
list(REMOVE_ITEM INTERNAL_GSERVER_HEADER
        extension/FcrCvmDataProvider.h)

macro(filter_test VAR_NAME)
    set(tmp)
    foreach(p IN LISTS ${VAR_NAME})
        if(NOT ${p} MATCHES ".*tests/.*")
             set(tmp ${p} ${tmp})
        endif()
    endforeach()
    set(${VAR_NAME} ${tmp})
endmacro()

filter_test(INTERNAL_GSERVER_HEADER)
filter_test(INTERNAL_GSERVER_SOURCES)

if(NOT WITH_GPU OR NOT WITH_PYTHON)
    list(REMOVE_ITEM INTERNAL_GSERVER_SOURCES
            dataproviders/ImageDataProvider.cpp
            dataproviders/ImageExtract.cpp)

    list(REMOVE_ITEM INTERNAL_GSERVER_HEADER
            dataproviders/ImageDataProvider.h
            dataproviders/ImageExtract.h)
endif()

if(NOT WITH_GPU)
    list(REMOVE_ITEM INTERNAL_GSERVER_SOURCES
            layers/PerturbationLayer.cpp)

    list(REMOVE_ITEM INTERNAL_GSERVER_HEADER
            layers/PerturbationLayer.cpp)
endif()

add_library(paddle_internal_gserver STATIC
            ${INTERNAL_GSERVER_SOURCES})
add_style_check_target(paddle_internal_gserver ${INTERNAL_GSERVER_SOURCES})
add_style_check_target(paddle_internal_gserver ${INTERNAL_GSERVER_HEADER})
add_dependencies(paddle_internal_gserver gen_proto_cpp)
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
