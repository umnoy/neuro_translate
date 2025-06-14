#include <iostream>
#include <string>
#include <vector>
#include <limits> // Для numeric_limits
#include <locale>   // Для работы с локалью
#include <clocale>  // Для std::setlocale

// Убедись, что у тебя подключены заголовочные файлы для Tokenizer и Translator
// Например:
#include "tokenizer/tokenizer.hpp"
#include "translator/traslator.hpp"


using namespace std;

// Предполагается, что классы Tokenizer и Translator определены в подключенных файлах

int main(int argc, char *argv[]) {

    // === НАСТРОЙКА ЛОКАЛИ ДЛЯ РУССКОГО ЯЗЫКА И UTF-8 ===
    // Попытка установить системную локаль.
    // В большинстве современных систем Linux системная локаль уже настроена на UTF-8.
    // "" означает использование локали из переменных окружения системы (LANG, LC_CTYPE и т.д.)
    try {
        std::locale::global(std::locale("")); // C++ способ
        // Или можно использовать C-стиль:
        // std::setlocale(LC_ALL, "");
        cout << "Локаль успешно установлена." << endl; // Отладочное сообщение
    } catch (const std::runtime_error& e) {
        cerr << "Ошибка установки локали: " << e.what() << endl;
        // Попробуем установить явную UTF-8 локаль, если системная не сработала
        try {
             std::locale::global(std::locale("en_US.UTF-8")); // Пример для UTF-8
             // Или std::setlocale(LC_ALL, "en_US.UTF-8");
             cout << "Локаль установлена на en_US.UTF-8." << endl; // Отладочное сообщение
        } catch (const std::runtime_error& e2) {
             cerr << "Не удалось установить локаль en_US.UTF-8: " << e2.what() << endl;
             cerr << "Возможны проблемы с кодировкой." << endl;
        }
    }
    // Для синхронизации C++ потоков с C-потоками (setlocale)
    std::ios_base::sync_with_stdio(false); // Может немного ускорить ввод/вывод
    // ===================================================


    int choice;
    cout << "Выберите опцию перевода:" << endl;
    cout << "1 - С английского на русский (en-ru)" << endl;
    cout << "2 - С русского на английский (ru-en)" << endl;
    cout << "Ваш выбор: ";
    cin >> choice;

    // Очищаем буфер ввода после cin >> int, чтобы getline работал корректно
    cin.ignore(numeric_limits<streamsize>::max(), '\n');


    // === Объявляем переменные для путей ДО if/else блока ===
    std::string vocab_path;
    std::string encoder_path;
    std::string decoder_path;
    // =======================================================

    // === Определяем пути в зависимости от выбора пользователя ===
    if (choice == 1) {
        // Пути для перевода en-ru
        vocab_path = "../models/opus-mt-en-ru/vocab.json";
        encoder_path = "../models/opus-mt-en-ru/encoder_model.onnx";
        decoder_path = "../models/opus-mt-en-ru/decoder_model.onnx";
        cout << "Выбран перевод en-ru." << endl;
    } else if (choice == 2) {
        // Пути для перевода ru-en - меняем 'en-ru' на 'ru-en' в путях
        vocab_path = "../models/opus-mt-ru-en/vocab.json";
        encoder_path = "../models/opus-mt-ru-en/encoder_model.onnx";
        decoder_path = "../models/opus-mt-ru-en/decoder_model.onnx";
        cout << "Выбран перевод ru-en." << endl;
    } else {
        // Обработка неверного ввода
        cerr << "Неверный выбор опции перевода. Завершение программы." << endl;
        return 1; // Выходим с кодом ошибки
    }
    // =============================================================

    // Объявляем и инициализируем другие параметры
    // Убедись, что эти значения (pad_token_id, eos_token_id) подходят для обеих моделей,
    // или определи их внутри if/else, если они зависят от модели.
    // Возможно, для русской модели нужен другой pad_token_id/eos_token_id!
    int pad_token_id = 62517; // Проверь это значение для ru-en модели!
    int eos_token_id = 0;   // Проверь это значение для ru-en модели!
    int max_length = 20;
    int beam_width = 3;


    // === СОЗДАЕМ ОБЪЕКТ TOKENIZER ПОСЛЕ ТОГО, КАК ПУТЬ К СЛОВАРЮ ОПРЕДЕЛЕН ===
    Tokenizer tokenizer(vocab_path);
    // ==========================================================================

    // === СОЗДАЕМ ОБЪЕКТ TRANSLATOR ===
    Translator translator(tokenizer, encoder_path, decoder_path, pad_token_id, eos_token_id, max_length, beam_width);
    // ================================

    // Временно закомментированный тестовый блок остается закомментированным
    /*
    {
        string sentence = "Hello, how are you?";
        vector<int64_t> ids = tokenizer.encode(sentence);
        string decoded = tokenizer.decode(ids);
        cout << "[TEST] Tokenize/Detokenize: " << sentence << " -> " << decoded << endl;
    }
    */


    // === Блок ввода текста и перевода пользователем ===
    string input_text;
    cout << "Введите текст для перевода (или 'exit' для выхода):" << endl;
    while (true) {
        cout << "> "; // Приглашение для ввода
        // getline должен теперь корректно читать многобайтовые символы
        getline(cin, input_text);

        if (input_text == "exit") {
            break;
        }
        // Проверка на пустой ввод
        if (input_text.empty()) {
            cout << "Введите текст." << endl;
            continue;
        }

        // Здесь происходит вызов метода run, который, вероятно, использует ONNX Runtime
        // Теперь input_text должна содержать корректные русские символы
        string result = translator.run(input_text);
        cout << "Переведено: " << result << endl;
    }
    // ===============================================


    return 0;
}
