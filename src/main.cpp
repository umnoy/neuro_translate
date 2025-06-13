#define NOMINMAX // !!! В САМОМ ВЕРХУ, ПЕРЕД ЛЮБЫМИ ВКЛЮЧЕНИЯМИ, чтобы избежать конфликта с std::max !!!

#include "nn/neural_translator.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <limits> // Для std::numeric_limits

#ifdef _WIN32
#include <windows.h> // Для SetConsoleOutputCP
#endif

// Определим путь для отладочных файлов в main.cpp или другом общем месте
const std::string C_PLUS_PLUS_DEBUG_DIR = "./debug_tensors";


int main() {
    // Настройка кодировки консоли для корректного отображения кириллицы
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    // std::cout << "\xEF\xBB\xBF"; // BOM для UTF-8 (опционально, может быть проблемой)
    #endif

    try {
        std::string model_base_dir = "C:/importantpapka/translator/models";

        std::string source_lang;
        std::string target_lang;
        std::string choice;

        std::cout << "Выберите направление перевода (en-ru или ru-en): ";
        std::cin >> choice;

        if (choice == "en-ru") {
            source_lang = "en";
            target_lang = "ru";
        } else if (choice == "ru-en") {
            source_lang = "ru";
            target_lang = "en";
        } else {
            std::cerr << "Неверный выбор. Доступны только 'en-ru' и 'ru-en'." << std::endl;
            return 1;
        }

        // Очищаем буфер ввода после std::cin для корректной работы std::getline
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        // Инициализация переводчика
        NeuralTranslator translator(model_base_dir, source_lang, target_lang);

        // Ввод текста для перевода
        std::cout << "Введите текст для перевода (" << source_lang << " -> " << target_lang << "): ";
        std::string input_text;
        std::getline(std::cin, input_text);

        if (input_text.empty()) {
            std::cerr << "Ошибка: пустой ввод" << std::endl;
            return 1;
        }

        // Выполнение перевода
        std::string translated_text = translator.translate(input_text);

        // Проверяем, вернул ли метод translate сообщение об ошибке
        if (translated_text.rfind("[Ошибка:", 0) == 0) {
             std::cerr << "Ошибка перевода: " << translated_text << std::endl;
             return 1;
        } else {
             std::cout << "Перевод: " << translated_text << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Исключение: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Неизвестное исключение." << std::endl;
        return 1;
    }

    return 0;
}