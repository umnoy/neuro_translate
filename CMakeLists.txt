cmake_minimum_required(VERSION 3.16)
project(translator LANGUAGES CXX)

# Устанавливаем стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Путь к ONNX Runtime
set(ONNXRUNTIME_ROOT "C:/importantpapka/onnxruntime-win-x64-1.17.1")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Подключаем поддиректории
add_subdirectory(src)
# Подключаем test только если папка существует и содержит CMakeLists.txt
if(EXISTS "${CMAKE_SOURCE_DIR}/test/CMakeLists.txt")
    add_subdirectory(test)
endif()