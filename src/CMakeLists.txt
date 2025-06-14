# Явно указываем исходные файлы
set(SOURCES
    main.cpp
    tokenizer/tokenizer.cpp
    translator/translator.cpp
)

# Создаём исполняемый файл
add_executable(translator ${SOURCES})

# Подключаем nlohmann/json
target_link_libraries(translator PRIVATE nlohmann_json::nlohmann_json)

# Подключаем ONNX Runtime
target_include_directories(translator PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_directories(translator PRIVATE ${ONNXRUNTIME_LIB_DIR})

# Ищем библиотеку ONNX Runtime
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}")
endif()

target_link_libraries(translator PRIVATE ${ONNXRUNTIME_LIBRARY})

# Указываем include-пути для исходников
target_include_directories(translator PRIVATE ${CMAKE_SOURCE_DIR}/src)