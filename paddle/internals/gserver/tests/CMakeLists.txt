# unittest for internal gserver package

############### test_FullContextLayer ####################
add_unittest(test_FullContextLayer
    test_FullContextLayer.cpp
    ../../../gserver/tests/TestUtil.cpp)

############## test_SelectiveFCLayer #####################
add_unittest_without_exec(test_SelectiveFCLayer
    test_SelectiveFCLayer.cpp
    ../../../gserver/tests/TestUtil.cpp)

add_test(NAME test_SelectiveFCLayer
    COMMAND .set_python_path.sh -d ${PROJ_ROOT}/python ${CMAKE_CURRENT_BINARY_DIR}/test_SelectiveFCLayer
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle)
