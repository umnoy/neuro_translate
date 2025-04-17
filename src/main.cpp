#include <iostream>
#include "translator.hpp"

int main() {
    Translator t;
    std::string input = "Привет, мир!";
    std::string output = t.translate(input);
    std::cout << "Translation: " << output << std::endl;
    return 0;
}
