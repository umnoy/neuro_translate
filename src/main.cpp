#include "nn/neural_translator.hpp"
#include <iostream>
#include <string>
#include <Windows.h>

int main() {

    std::cout << "Text: " << std::flush;
    std::string input;
    std::getline(std::cin, input);

    try {
        // Временные пути (замени на реальные, когда будут доступны)
        std::string model_path = "C:\\importantpapka\\translator\\importing\\onnx\\model.onnx"; // Укажи путь к модели
        std::string tokenizer_dir ="C:\\importantpapka\\translator\\importing\\onnxmodel.onnx"; // Укажи путь к токенизатору
        NeuralTranslator translator(model_path, tokenizer_dir);
        std::string result = translator.translate(input);
        std::cout << "Translated: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}