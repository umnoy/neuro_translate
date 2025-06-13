# В поддиректории src, CMake уже "знает" переменные и директории, 
# определенные в родительском CMakeLists.txt (например, ONNXRUNTIME_ROOT)

# Добавляем статическую библиотеку neural_translator
add_library(neural_translator STATIC nn/neural_translator.cpp)

# Указываем директории для включения заголовков. 
# Используем PUBLIC, так как заголовки ONNX Runtime и SentencePiece 
# используются в neural_translator.hpp и нужны для компиляции translator_app.
target_include_directories(neural_translator PUBLIC 
    ${ONNXRUNTIME_ROOT}/include 
    ${SENTENCEPIECE_INCLUDE_DIR} # Используем переменную из корневого CMakeLists
    # Если твои собственные заголовки (.hpp) лежат где-то еще и 
    # используются в neural_translator.cpp, добавь их сюда PRIVATELY
    # Например: PRIVATE ${CMAKE_SOURCE_DIR}/../include 
)

# Привязываем библиотеки к статической библиотеке neural_translator.
# Обычно библиотеки линкуются PRIVATE к таргету, который их использует.
# Обрати внимание: имя библиотеки ONNX Runtime - onnxruntime.lib
target_link_libraries(neural_translator PRIVATE 
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib # Полный путь к файлу .lib
    sentencepiece # Имя библиотеки SentencePiece, как она называется в твоей системе (например, sentencepiece.lib -> "sentencepiece")
)


# Добавляем исполняемый файл translator_app
add_executable(translator_app main.cpp)

# Привязываем библиотеки к исполняемому файлу translator_app.
# translator_app зависит от neural_translator.
# Он также нуждается в библиотеках, которые использует neural_translator (ONNX Runtime, SentencePiece).
# Хотя PUBLIC include directories из neural_translator должны работать,
# линковку внешних библиотек (особенно .lib) часто нужно повторить для исполняемого файла.
target_link_libraries(translator_app 
    PRIVATE # Связываем с translator_app приватно
        neural_translator # Сначала наша библиотека
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib # Затем ONNX Runtime
        sentencepiece # Затем SentencePiece
)

# TODO: Проверь, что имя библиотеки SentencePiece 'sentencepiece' правильное для твоей системы.
# В зависимости от того, как ты собрал или получил SentencePiece, имя файла .lib 
# может быть другим (например, libsentencepiece.lib или просто sentencepiece.lib).
# В CMake target_link_libraries обычно указывается логическое имя библиотеки (без префиксов типа lib и суффиксов .lib/.dll), 
# но для файлов .lib иногда надежнее указывать полный путь.
# Если "sentencepiece" не работает, попробуй полный путь: ${SENTENCEPIECE_LIB_DIR}/sentencepiece.lib
