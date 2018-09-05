# The utilities for paddle

file(GLOB PARAMETERS_HEADERS . *.h)
file(GLOB PARAMETERS_SOURCES . *.cpp)

add_library(paddle_internal_parameter STATIC
            ${PARAMETERS_SOURCES})
#add_style_check_target(paddle_internal_parameter ${PARAMETERS_SOURCES})
#add_style_check_target(paddle_internal_parameter ${PARAMETERS_HEADERS})
add_dependencies(paddle_internal_parameter gen_proto_cpp)
