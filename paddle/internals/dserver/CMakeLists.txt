set(DSERVER_HEADERS
    DataClient.h
    DataServer.h)

set(DSERVER_SOURCES
    DataClient.cpp
    DataServer.cpp)


if(NOT WITH_GPU)
    add_library(paddle_dserver_lib STATIC
        ${DSERVER_SOURCES})    
else()
    add_paddle_culib(paddle_dserver_lib STATIC
        ${DSERVER_SOURCES})
endif()



add_style_check_target(paddle_dserver_lib ${DSERVER_SOURCES})
add_style_check_target(paddle_dserver_lib ${DSERVER_HEADERS})
add_dependencies(paddle_dserver_lib gen_proto_cpp)  # depends

add_executable(paddle_dserver
        DataServerMain.cpp)

link_paddle_exe(paddle_dserver)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
