cmake_minimum_required(VERSION 3.10)

project(neural_net CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf CONFIG REQUIRED)


protobuf_generate_cpp(NET_SRCS NET_HDRS neural_net.proto)

set(NET_FILES 
main.cpp
neuronet.cpp
neuronet.h
)



add_executable(neural ${NET_SRCS} ${NET_HDRS} ${NET_FILES})

target_include_directories(neural PUBLIC ${Protobuf_INCLUDE_DIRS})
target_include_directories(neural PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


string(REPLACE "protobuf.lib" "protobufd.lib" "Protobuf_LIBRARY_DEBUG" "${Protobuf_LIBRARY_DEBUG}")
string(REPLACE "protobuf.a" "protobufd.a" "Protobuf_LIBRARY_DEBUG" "${Protobuf_LIBRARY_DEBUG}")



target_link_libraries(neural "$<IF:$<CONFIG:Debug>,${Protobuf_LIBRARY_DEBUG},${Protobuf_LIBRARY}>")
