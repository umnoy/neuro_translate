#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>  // Для std::unique_ptr
#include <stdexcept> // Для std::runtime_error

// ONNX Runtime и SentencePiece
#include <onnxruntime_cxx_api.h>
#include <sentencepiece/sentencepiece_processor.h>

// OrtAllocatorDeleter полностью удален, так как OrtAllocator* не требует ручного освобождения

class NeuralTranslator {
public:
    // Константа для отладки, теперь публичная
    static constexpr bool DEBUG_MODE = true;           // Флаг отладки

    // Конструктор: принимает директорию моделей и языки перевода
    NeuralTranslator(const std::string& model_base_dir, const std::string& source_lang, const std::string& target_lang);

    // Метод перевода текста
    std::string translate(const std::string& input_text);

private:
    Ort::Env env_; // ONNX Runtime окружение
    Ort::MemoryInfo cpu_memory_info_; // Информация о памяти CPU

    // ИЗМЕНЕНО: default_allocator_ теперь обычный сырой указатель OrtAllocator*
    // Получается через env_.GetAllocator() и не требует ручного освобождения.
    OrtAllocator* default_allocator_ = nullptr;

    // ONNX Runtime сессии для разных частей модели
    std::unique_ptr<Ort::Session> encoder_session_ = nullptr;
    std::unique_ptr<Ort::Session> decoder_session_ = nullptr; // Для первого шага декодирования (без past)
    std::unique_ptr<Ort::Session> decoder_with_past_session_ = nullptr; // Для последующих шагов декодирования (с past)

    // SentencePiece процессоры
    sentencepiece::SentencePieceProcessor source_sp_processor_; // Для токенизации входного языка
    sentencepiece::SentencePieceProcessor target_sp_processor_; // Для декодирования выходного языка

    std::vector<const char*> decoder_current_input_names_chars;
    std::vector<const char*> decoder_current_output_names_chars;

    // ID специальных токенов
    int bos_token_id_ = 0;
    int eos_token_id_ = 1;
    int pad_token_id_ = 62517;
    int vocab_size_ = 62518;

    // Языки перевода
    std::string source_lang_;
    std::string target_lang_;

    // Константы
    static constexpr int MAX_TRANSLATION_LENGTH = 50;// Максимальная длина генерируемой последовательности

    // Вспомогательная функция для загрузки модели ONNX
    void LoadModel(const std::string& model_path, std::unique_ptr<Ort::Session>& session);

    // Токенизация входного текста (source_lang)
    std::vector<int64_t> TokenizeInput(const std::string& text);

    // Декодирование выходных ID токенов (target_lang)
    std::string DecodeOutput(const std::vector<int64_t>& token_ids);

    // Вспомогательная функция для получения индекса с максимальным значением
    int get_max_index(const float* data, size_t size);

    // Вспомогательные функции для отладки (сохранение тензоров в файлы)
    void SaveInt64Vector(const std::vector<int64_t>& data, const std::string& filename_suffix);
    void SaveFloatVector(const std::vector<float>& data, const std::string& filename_suffix);
    void SaveFloatTensor(const float* data, const std::vector<int64_t>& shape, const std::string& filename_suffix);
    void LogTensorInfo(const Ort::Value& tensor, const std::string& name, const std::string& type, int step);
};

// Глобальная константа для директории отладки
extern const std::string C_PLUS_PLUS_DEBUG_DIR;