#pragma once
#include <string>
#include <onnxruntime_cxx_api.h>

class NeuralTranslator {
public:
    NeuralTranslator(const std::string& model_path, const std::string& tokenizer_dir);
    std::string translate(const std::string& input);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;
    std::string tokenizer_path_;
};
