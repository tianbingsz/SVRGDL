#################### test_internal_config_parser #########################
add_test(NAME test_internal_config_parser
  COMMAND ${PROJ_ROOT}/paddle/.set_python_path.sh -d ${PROJ_ROOT}/python/
        python ${PROJ_ROOT}/paddle/internals/trainer/tests/config_parser_test.py
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle/)
