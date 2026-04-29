cmake_minimum_required(VERSION 3.14)

find_package(Lua 5.4 QUIET)

if (NOT LUA_FOUND)
    message(STATUS "llama-lua: Lua 5.4 not found, skipping llama-lua tool")
    return()
endif()

set(TARGET llama-lua)
add_executable(${TARGET} lua-runner.cpp)
target_include_directories(${TARGET} PRIVATE ${LUA_INCLUDE_DIR})
target_link_libraries(${TARGET} PRIVATE llama ${LUA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
    install(DIRECTORY scripts/ DESTINATION share/llama-lua/scripts FILES_MATCHING PATTERN "*.lua")
endif()
