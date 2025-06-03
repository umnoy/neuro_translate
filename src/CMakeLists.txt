# Создаём статическую библиотеку neural_translator
add_library(neural_translator STATIC nn/neural_translator.cpp)
target_include_directories(neural_translator PRIVATE ${ONNXRUNTIME_ROOT}/include)

# Создаём исполняемый файл translator_app
add_executable(translator_app main.cpp)
target_include_directories(translator_app PRIVATE ${ONNXRUNTIME_ROOT}/include)

# Компонуем translator_app
target_link_libraries(translator_app
    PRIVATE
        neural_translator
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
)