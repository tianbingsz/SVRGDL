file(GLOB INTERNAL_OWLQN_HEADERS . *.h)
file(GLOB INTERNAL_OWLQN_SOURCES . *.cpp)

add_library(paddle_internal_owlqn STATIC
            ${INTERNAL_OWLQN_SOURCES})
add_style_check_target(paddle_internal_owlqn ${INTERNAL_OWLQN_SOURCES})
add_style_check_target(paddle_internal_owlqn ${INTERNAL_OWLQN_HEADERS})
add_dependencies(paddle_internal_owlqn gen_proto_cpp)
