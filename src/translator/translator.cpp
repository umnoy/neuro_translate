#include "neural_translator.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <numeric>
#include <fstream>
#include <sstream>
#include <codecvt>
#include <locale>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

// Глобальная константа для директории отладки
//const std::string C_PLUS_PLUS_DEBUG_DIR = "./debug_tensors";

// Вспомогательная функция для создания директории отладки
static void CreateDebugDir() {
    if (C_PLUS_PLUS_DEBUG_DIR.empty()) return;

    std::filesystem::path debug_path(C_PLUS_PLUS_DEBUG_DIR);
    if (!std::filesystem::exists(debug_path)) {
        try {
            std::filesystem::create_directories(debug_path);
            if (NeuralTranslator::DEBUG_MODE) {
                std::cout << "Created debug directory: " << C_PLUS_PLUS_DEBUG_DIR << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating debug directory: " << e.what() << std::endl;
        }
    }
}

// Вспомогательные функции для отладки
void NeuralTranslator::SaveInt64Vector(const std::vector<int64_t>& data, const std::string& filename_suffix) {
    if (!DEBUG_MODE) return;
    CreateDebugDir();
    std::ofstream ofs(C_PLUS_PLUS_DEBUG_DIR + "/int64_" + filename_suffix + ".txt");
    if (!ofs.is_open()) {
        std::cerr << "Error opening file for saving int64 vector: " << C_PLUS_PLUS_DEBUG_DIR + "/int64_" + filename_suffix + ".txt" << std::endl;
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        ofs << data[i] << (i == data.size() - 1 ? "" : ", ");
    }
    ofs << std::endl;
    ofs.close();
}

void NeuralTranslator::SaveFloatVector(const std::vector<float>& data, const std::string& filename_suffix) {
    if (!DEBUG_MODE) return;
    CreateDebugDir();
    std::ofstream ofs(C_PLUS_PLUS_DEBUG_DIR + "/float_" + filename_suffix + ".txt");
    if (!ofs.is_open()) {
        std::cerr << "Error opening file for saving float vector: " << C_PLUS_PLUS_DEBUG_DIR + "/float_" + filename_suffix + ".txt" << std::endl;
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        ofs << data[i] << (i == data.size() - 1 ? "" : ", ");
    }
    ofs << std::endl;
    ofs.close();
}

void NeuralTranslator::SaveFloatTensor(const float* data, const std::vector<int64_t>& shape, const std::string& filename_suffix) {
    if (!DEBUG_MODE) return;
    CreateDebugDir();
    std::ofstream ofs(C_PLUS_PLUS_DEBUG_DIR + "/tensor_" + filename_suffix + ".txt");
    if (!ofs.is_open()) {
        std::cerr << "Error opening file for saving float tensor: " << C_PLUS_PLUS_DEBUG_DIR + "/tensor_" + filename_suffix + ".txt" << std::endl;
        return;
    }
    ofs << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        ofs << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ofs << "]" << std::endl;

    size_t total_elements = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    for (size_t i = 0; i < total_elements; ++i) {
        ofs << data[i] << (i == total_elements - 1 ? "" : ", ");
    }
    ofs << std::endl;
    ofs.close();
}

int NeuralTranslator::get_max_index(const float* data, size_t size) {
    if (size == 0) return -1;
    int max_idx = 0;
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > data[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

void NeuralTranslator::LoadModel(const std::string& model_path, std::unique_ptr<Ort::Session>& session) {
    if (DEBUG_MODE) {
        std::cout << "Loading model: " << model_path << std::endl;
    }
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency() / 2 > 0 ? std::thread::hardware_concurrency() / 2 : 1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring wide_model_path = converter.from_bytes(model_path);

    session = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options);

    if (DEBUG_MODE) {
        std::cout << "Model loaded successfully: " << model_path << std::endl;
    }
}

NeuralTranslator::NeuralTranslator(const std::string& model_base_dir, const std::string& source_lang, const std::string& target_lang)
    : env_(ORT_LOGGING_LEVEL_WARNING, "NeuralTranslator"),
      cpu_memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      source_lang_(source_lang),
      target_lang_(target_lang)
{
    CreateDebugDir();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    default_allocator_ = Ort::AllocatorWithDefaultOptions();

    if (default_allocator_ == nullptr) {
        throw std::runtime_error("Failed to get default ONNX Runtime allocator (returned nullptr).");
    }

    std::string model_folder_name;
    std::string source_sp_model_path;
    std::string target_sp_model_path;

    if (source_lang_ == "en" && target_lang_ == "ru") {
        model_folder_name = "opus-mt-en-ru";
    } else if (source_lang_ == "ru" && target_lang_ == "en") {
        model_folder_name = "opus-mt-ru-en";
    } else {
        throw std::runtime_error("Unsupported language pair: " + source_lang_ + " to " + target_lang_);
    }

    std::string current_model_dir = model_base_dir + "/" + model_folder_name;
    source_sp_model_path = current_model_dir + "/source.spm";
    target_sp_model_path = current_model_dir + "/target.spm";

    LoadModel(current_model_dir + "/encoder_model.onnx", encoder_session_);
    LoadModel(current_model_dir + "/decoder_model.onnx", decoder_session_);
    LoadModel(current_model_dir + "/decoder_with_past_model.onnx", decoder_with_past_session_);

    auto sp_status_source = source_sp_processor_.Load(source_sp_model_path);

    // bos_token_id_ = source_sp_processor_.bos_id();
    // eos_token_id_ = source_sp_processor_.eos_id();
    // pad_token_id_ = source_sp_processor_.pad_id();
    // vocab_size_ = source_sp_processor_.GetPieceSize();

    if (!sp_status_source.ok()) {
        throw std::runtime_error("Failed to load source SentencePiece model: " + source_sp_model_path + " Error: " + sp_status_source.ToString());
    }
    if (DEBUG_MODE) {
        std::cout << "Source SentencePiece model loaded: " << source_sp_model_path << std::endl;
    }

    auto sp_status_target = target_sp_processor_.Load(target_sp_model_path);
    if (!sp_status_target.ok()) {
        throw std::runtime_error("Failed to load target SentencePiece model: " + target_sp_model_path + " Error: " + sp_status_target.ToString());
    }
    if (DEBUG_MODE) {
        std::cout << "Target SentencePiece model loaded: " << target_sp_model_path << std::endl;
    }



    if (DEBUG_MODE) {
        std::cout << "Special token IDs: BOS=" << bos_token_id_
                  << ", EOS=" << eos_token_id_
                  << ", PAD=" << pad_token_id_
                  << ", Vocab_size= " << vocab_size_ << std::endl;
    }
}

std::vector<int64_t> NeuralTranslator::TokenizeInput(const std::string& text) {
    std::vector<int64_t> ids;
    std::vector<int> temp_ids_int = source_sp_processor_.EncodeAsIds(text);

    std::vector<int64_t> padded_ids;
    padded_ids.reserve(temp_ids_int.size() + 2);
    padded_ids.push_back(bos_token_id_);
    for (int id : temp_ids_int) {
        padded_ids.push_back(static_cast<int64_t>(id));
    }
    padded_ids.push_back(eos_token_id_);
    return padded_ids;
}

std::string NeuralTranslator::DecodeOutput(const std::vector<int64_t>& token_ids) {
    std::vector<int> filtered_ids;
    for (int64_t id : token_ids) {
        if (id != bos_token_id_ && id != eos_token_id_ && id != pad_token_id_) {
            filtered_ids.push_back(static_cast<int>(id));
        }
    }

    std::string decoded_text;
    auto sp_status = target_sp_processor_.Decode(filtered_ids, &decoded_text);
    if (!sp_status.ok()) {
        throw std::runtime_error("SentencePiece decoding failed: " + sp_status.ToString());
    }
    return decoded_text;
}

std::string NeuralTranslator::translate(const std::string& input_text) {
    try {
        // 1. Токенизация входного текста
        
        std::vector<int64_t> encoder_input_ids = TokenizeInput(input_text);
        if (DEBUG_MODE) {
            SaveInt64Vector(encoder_input_ids, "encoder_input_ids");
            std::cout << "Encoder input IDs size: " << encoder_input_ids.size() << std::endl;
        }

        std::cout << "[LOG] Source tokens: ";
        for (auto id : encoder_input_ids) std::cout << id << " ";
        std::cout << std::endl;

        // 2. Подготовка входных тензоров для энкодера
        std::vector<int64_t> encoder_input_shape = {1, static_cast<int64_t>(encoder_input_ids.size())};

        Ort::Value encoder_input_ids_tensor = Ort::Value::CreateTensor(
            cpu_memory_info_,
            encoder_input_ids.data(),
            encoder_input_ids.size() * sizeof(int64_t),
            encoder_input_shape.data(),
            encoder_input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        );

        std::vector<int64_t> encoder_attention_mask_data(encoder_input_ids.size(), 1);
        std::vector<int64_t> encoder_attention_mask_shape = {1, static_cast<int64_t>(encoder_input_ids.size())};
        Ort::Value encoder_attention_mask_tensor = Ort::Value::CreateTensor(
            cpu_memory_info_,
            encoder_attention_mask_data.data(),
            encoder_attention_mask_data.size() * sizeof(int64_t),
            encoder_attention_mask_shape.data(),
            encoder_attention_mask_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        );

        std::vector<const char*> encoder_input_names = {"input_ids", "attention_mask"};
        std::vector<const char*> encoder_output_names = {"last_hidden_state"};
        std::vector<Ort::Value> encoder_run_inputs;
        encoder_run_inputs.push_back(std::move(encoder_input_ids_tensor));
        encoder_run_inputs.push_back(std::move(encoder_attention_mask_tensor));

        // 3. Выполнение энкодера
        if (DEBUG_MODE) {
            std::cout << "Running encoder session..." << std::endl;
        }
        auto encoder_outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr},
            encoder_input_names.data(),
            encoder_run_inputs.data(),
            encoder_run_inputs.size(),
            encoder_output_names.data(),
            encoder_output_names.size()
        );
        if (DEBUG_MODE) {
            std::cout << "Encoder session finished." << std::endl;
        }

        // Получаем скрытые состояния энкодера
        Ort::Value encoder_hidden_states_output_value = std::move(encoder_outputs.front());
        const auto& encoder_hidden_states_shape = encoder_hidden_states_output_value.GetTensorTypeAndShapeInfo().GetShape();
        const float* encoder_hidden_states_data = encoder_hidden_states_output_value.GetTensorData<float>();
        if (DEBUG_MODE) {
            std::cout << "Encoder hidden states shape: [" << encoder_hidden_states_shape[0] << ", "
                      << encoder_hidden_states_shape[1] << ", " << encoder_hidden_states_shape[2] << "]" << std::endl;
            SaveFloatTensor(encoder_hidden_states_data, encoder_hidden_states_shape, "encoder_hidden_states");
        }

        // Создаем тензоры для повторного использования
        std::vector<Ort::Value> persistent_encoder_tensors; // Для хранения неизменяемых тензоров
        persistent_encoder_tensors.push_back(Ort::Value::CreateTensor(
            cpu_memory_info_,
            const_cast<float*>(encoder_hidden_states_data),
            encoder_hidden_states_shape[0] * encoder_hidden_states_shape[1] * encoder_hidden_states_shape[2] * sizeof(float),
            encoder_hidden_states_shape.data(),
            encoder_hidden_states_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        ));

        persistent_encoder_tensors.push_back(Ort::Value::CreateTensor(
            cpu_memory_info_,
            const_cast<int64_t*>(encoder_attention_mask_data.data()),
            encoder_attention_mask_data.size() * sizeof(int64_t),
            encoder_attention_mask_shape.data(),
            encoder_attention_mask_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        ));

        // 4. Цикл авторегрессивного декодирования
        std::vector<int64_t> generated_token_ids;
        generated_token_ids.push_back(bos_token_id_);
        int64_t next_token_id = bos_token_id_;
        std::vector<Ort::Value> past_key_values_for_next_step;

        std::cout << "[LOG] Decoder start tokens: ";
        for (auto id : generated_token_ids) std::cout << id << " ";
        std::cout << std::endl;

        for (int i = 0; i < MAX_TRANSLATION_LENGTH; ++i) {
            std::vector<Ort::Value> decoder_inputs;

            if (DEBUG_MODE) {
                std::cout << "Decoding step: " << i << std::endl;
            }

            if (i == 0) {
                // --- Для decoder_model.onnx (первый шаг) ---
                std::vector<int64_t> initial_decoder_input_ids = {bos_token_id_};
                std::vector<int64_t> initial_decoder_input_shape = {1, 1};
                Ort::Value initial_decoder_input_ids_tensor = Ort::Value::CreateTensor(
                    cpu_memory_info_,
                    initial_decoder_input_ids.data(),
                    initial_decoder_input_ids.size() * sizeof(int64_t),
                    initial_decoder_input_shape.data(),
                    initial_decoder_input_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                );

                decoder_inputs.push_back(std::move(persistent_encoder_tensors[1])); // encoder_attention_mask
                decoder_inputs.push_back(std::move(initial_decoder_input_ids_tensor)); // input_ids
                decoder_inputs.push_back(std::move(persistent_encoder_tensors[0])); // encoder_hidden_states

                decoder_current_input_names_chars = {
                    "encoder_attention_mask",
                    "input_ids",
                    "encoder_hidden_states"
                };

                decoder_current_output_names_chars = {
                    "logits",
                    "present.0.decoder.key", "present.0.decoder.value",
                    "present.0.encoder.key", "present.0.encoder.value",
                    "present.1.decoder.key", "present.1.decoder.value",
                    "present.1.encoder.key", "present.1.encoder.value",
                    "present.2.decoder.key", "present.2.decoder.value",
                    "present.2.encoder.key", "present.2.encoder.value",
                    "present.3.decoder.key", "present.3.decoder.value",
                    "present.3.encoder.key", "present.3.encoder.value",
                    "present.4.decoder.key", "present.4.decoder.value",
                    "present.4.encoder.key", "present.4.encoder.value",
                    "present.5.decoder.key", "present.5.decoder.value",
                    "present.5.encoder.key", "present.5.encoder.value"
                };
            } else {
                // --- Для decoder_with_past_model.onnx (последующие шаги) ---
                std::vector<int64_t> single_token_input_ids = {next_token_id};
                std::vector<int64_t> single_token_input_shape = {1, 1};
                Ort::Value single_token_input_tensor = Ort::Value::CreateTensor(
                    cpu_memory_info_,
                    single_token_input_ids.data(),
                    single_token_input_ids.size() * sizeof(int64_t),
                    single_token_input_shape.data(),
                    single_token_input_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                );

                // Восстанавливаем encoder_hidden_states и encoder_attention_mask из persistent_encoder_tensors
                decoder_inputs.push_back(Ort::Value::CreateTensor(
                    cpu_memory_info_,
                    const_cast<int64_t*>(encoder_attention_mask_data.data()),
                    encoder_attention_mask_data.size() * sizeof(int64_t),
                    encoder_attention_mask_shape.data(),
                    encoder_attention_mask_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                ));

                decoder_inputs.push_back(std::move(single_token_input_tensor));

                // Добавляем past_key_values (все 24 тензора)
                for (size_t j = 0; j < past_key_values_for_next_step.size(); ++j) {
                    // Копируем тензоры вместо перемещения, чтобы сохранить их для следующего шага
                    Ort::Value copied_tensor = Ort::Value::CreateTensor(
                        cpu_memory_info_,
                        past_key_values_for_next_step[j].GetTensorMutableData<float>(),
                        past_key_values_for_next_step[j].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float),
                        past_key_values_for_next_step[j].GetTensorTypeAndShapeInfo().GetShape().data(),
                        past_key_values_for_next_step[j].GetTensorTypeAndShapeInfo().GetShape().size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                    );
                    decoder_inputs.push_back(std::move(copied_tensor));
                }

                decoder_current_input_names_chars = {
                    "encoder_attention_mask",
                    "input_ids",
                    "past_key_values.0.decoder.key", "past_key_values.0.decoder.value",
                    "past_key_values.0.encoder.key", "past_key_values.0.encoder.value",
                    "past_key_values.1.decoder.key", "past_key_values.1.decoder.value",
                    "past_key_values.1.encoder.key", "past_key_values.1.encoder.value",
                    "past_key_values.2.decoder.key", "past_key_values.2.decoder.value",
                    "past_key_values.2.encoder.key", "past_key_values.2.encoder.value",
                    "past_key_values.3.decoder.key", "past_key_values.3.decoder.value",
                    "past_key_values.3.encoder.key", "past_key_values.3.encoder.value",
                    "past_key_values.4.decoder.key", "past_key_values.4.decoder.value",
                    "past_key_values.4.encoder.key", "past_key_values.4.encoder.value",
                    "past_key_values.5.decoder.key", "past_key_values.5.decoder.value",
                    "past_key_values.5.encoder.key", "past_key_values.5.encoder.value"
                };

                decoder_current_output_names_chars = {
                    "logits",
                    "present.0.decoder.key", "present.0.decoder.value",
                    "present.1.decoder.key", "present.1.decoder.value",
                    "present.2.decoder.key", "present.2.decoder.value",
                    "present.3.decoder.key", "present.3.decoder.value",
                    "present.4.decoder.key", "present.4.decoder.value",
                    "present.5.decoder.key", "present.5.decoder.value"
                };
            }

            Ort::Session* current_decoder_session = (i == 0) ? decoder_session_.get() : decoder_with_past_session_.get();

            auto decoder_outputs = current_decoder_session->Run(
                Ort::RunOptions{nullptr},
                decoder_current_input_names_chars.data(),
                decoder_inputs.data(),
                decoder_inputs.size(),
                decoder_current_output_names_chars.data(),
                decoder_current_output_names_chars.size()
            );

            Ort::Value& logits_tensor = decoder_outputs.front();
            const auto& logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
            const float* logits_data = logits_tensor.GetTensorData<float>();

            const float* current_logits_data = logits_data;
            if (logits_shape.size() == 3 && logits_shape[1] > 1) {
                current_logits_data = logits_data + ((logits_shape[1] - 1) * logits_shape[2]);
            }

            next_token_id = get_max_index(current_logits_data, vocab_size_);
            generated_token_ids.push_back(next_token_id);
            if (DEBUG_MODE) {
                SaveInt64Vector({next_token_id}, "generated_token_id_step_" + std::to_string(i));
                SaveFloatTensor(current_logits_data, {1, 1, static_cast<int64_t>(vocab_size_)}, "logits_step_" + std::to_string(i));
            }

            std::cout << "[LOG] Decoder step " << i << ", next token: " << next_token_id << std::endl;

            if (next_token_id == eos_token_id_ || generated_token_ids.size() >= MAX_TRANSLATION_LENGTH) {
                if (DEBUG_MODE) {
                    std::cout << "EOS token generated or max length reached. Stopping decoding." << std::endl;
                }
                break;
            }

            // Обновление past_key_values_for_next_step
            if (i == 0) {
                past_key_values_for_next_step.clear();
                for (size_t j = 1; j < decoder_outputs.size(); ++j) {
                    past_key_values_for_next_step.push_back(std::move(decoder_outputs[j]));
                }
            } else {
                if (past_key_values_for_next_step.size() != 24) {
                    throw std::runtime_error("Unexpected size of past_key_values_for_next_step before update: " +
                                             std::to_string(past_key_values_for_next_step.size()));
                }

                for (size_t k = 0; k < 6; ++k) {
                    past_key_values_for_next_step[k * 4] = std::move(decoder_outputs[1 + k * 2]);      // decoder.key
                    past_key_values_for_next_step[k * 4 + 1] = std::move(decoder_outputs[1 + k * 2 + 1]); // decoder.value
                    // encoder.key (k*4 + 2) и encoder.value (k*4 + 3) остаются неизменными
                }
            }
            if (DEBUG_MODE) {
                std::cout << "Number of past_key_values for next step: " << past_key_values_for_next_step.size() << std::endl;
            }
        }

        std::cout << "[LOG] Target tokens: ";
        for (auto id : generated_token_ids) std::cout << id << " ";
    std::cout << std::endl;

        std::string translated_text = DecodeOutput(generated_token_ids);
        if (DEBUG_MODE) {
            std::cout << "Final translated text: '" << translated_text << "'" << std::endl;
        }
        return translated_text;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception during translation: " << e.what() << std::endl;
        return "[Ошибка: ONNX Runtime: " + std::string(e.what()) + "]";
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception during translation: " << e.what() << std::endl;
        return "[Ошибка: Стандартное исключение: " + std::string(e.what()) + "]";
    } catch (...) {
        std::cerr << "Unknown Exception during translation." << std::endl;
        return "[Ошибка: Неизвестное исключение]";
    }
}