cmake_minimum_required(VERSION 3.16)
project(translator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Укажи здесь реальный путь к твоей установке ONNX Runtime
set(ONNXRUNTIME_ROOT "C:/importantpapka/onnxruntime-win-x64-1.17.1")
# Убедись, что эти директории существуют
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
    message(FATAL_ERROR "ONNX Runtime include directory not found: ${ONNXRUNTIME_ROOT}/include")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib")
    message(FATAL_ERROR "ONNX Runtime lib directory not found: ${ONNXRUNTIME_ROOT}/lib")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# SentencePiece
# Укажи здесь реальный путь к заголовкам и библиотеке SentencePiece
set(SENTENCEPIECE_INCLUDE_DIR "C:/importantpapka/translator/include")
set(SENTENCEPIECE_LIB_DIR "C:/importantpapka/translator/libs")

if(NOT EXISTS "${SENTENCEPIECE_INCLUDE_DIR}")
     message(FATAL_ERROR "SentencePiece include directory not found: ${SENTENCEPIECE_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${SENTENCEPIECE_LIB_DIR}")
     message(FATAL_ERROR "SentencePiece lib directory not found: ${SENTENCEPIECE_LIB_DIR}")
endif()


include_directories(${SENTENCEPIECE_INCLUDE_DIR})
link_directories(${SENTENCEPIECE_LIB_DIR})


add_subdirectory(src)

# Если у тебя есть папка test с CMakeLists.txt, оставь этот блок
if(EXISTS "${CMAKE_SOURCE_DIR}/test/CMakeLists.txt")
    add_subdirectory(test)
endif()
