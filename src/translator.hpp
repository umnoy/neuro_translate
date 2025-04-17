#pragma once
#include <string>

class Translator{
    public:
    Translator();
    std::string translate(const std::string& input_text);
};