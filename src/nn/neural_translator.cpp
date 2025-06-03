#include "neural_translator.hpp"
#include <iostream>
#include <codecvt>
#include <locale>

NeuralTranslator::NeuralTranslator(const std::string& model_path, const std::string& tokenizer_dir)
    : env_(ORT_LOGGING_LEVEL_WARNING, "translator")
{
    std::cout << "Инициализация NeuralTranslator с model_path: " << model_path << std::endl;

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        // Преобразование std::string в std::wstring
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::wstring wide_model_path = converter.from_bytes(model_path);

        // Используем широкую строку для пути к модели
        session_ = Ort::Session(env_, wide_model_path.c_str(), session_options_);
        std::cout << "Модель успешно загружена" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Ошибка загрузки модели ONNX: " << e.what() << std::endl;
        throw;
    }

    tokenizer_path_ = tokenizer_dir;
}

std::string NeuralTranslator::translate(const std::string& input) {
    std::cout << "NeuralTranslator::translate called with: " << input << std::endl;
    return "[Translated (ONNX placeholder)]: " + input;
}