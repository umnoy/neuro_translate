cmake_minimum_required(VERSION 3.16)
project(translator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Пути к ONNX Runtime
set(ONNXRUNTIME_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# Проверка существования директорий
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "ONNX Runtime include directory not found: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}")
    message(FATAL_ERROR "ONNX Runtime lib directory not found: ${ONNXRUNTIME_LIB_DIR}")
endif()

# Подключаем nlohmann/json
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Подключаем подкаталог src
add_subdirectory(src)

# Включаем подробный вывод для диагностики
set(CMAKE_VERBOSE_MAKEFILE ON)