#include <iostream>
#include "translator.hpp"

int main() {
    Translator t;
    std::string input;
    std::cout << "Введите текст: ";
    std::cin >> input;
    std::string output = t.translate(input);
    std::cout << "Translation: " << output << std::endl;
    return 0;
}
