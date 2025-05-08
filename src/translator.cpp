#include "translator.hpp"
#include <unordered_map>

Translator::Translator() {
    dictionary = {
        {"привет", "hello"},
        {"мир", "world"},
        {"как", "how"},
        {"дела", "are you"},
    };
}

std::string Translator::translate(const std::string& input_text) {
    std::stringstream ss(input_text);
    std::string word;
    std::string result;

    while (ss >> word) {
        // Преобразуем к нижнему регистру (опционально)
        std::string lower = word;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (dictionary.count(lower)) {
            result += dictionary[lower] + " ";
        } else {
            result += word + " ";
        }
    }

    return result;
}