macro(add_paddle_exe TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  add_style_check_target(${TARGET_NAME} ${ARGN})
  link_paddle_exe(${TARGET_NAME})
endmacro()

add_paddle_exe(paddle_gen_sequence
    GenerateSequence.cpp)

add_paddle_exe(paddle_gen_answer
    GenerateAnswer.cpp)
  
if(WITH_TESTING)
    add_subdirectory(tests)
endif()
