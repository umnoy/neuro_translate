#pragma once
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>

class Translator {
public:
    Translator();
    std::string translate(const std::string& input_text);
private:
    std::unordered_map<std::string, std::string> dictionary;
};
