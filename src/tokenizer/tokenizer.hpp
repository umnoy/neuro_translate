#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
  Tokenizer(const std::string &vocab_path);

  std::vector<int64_t> encode(const std::string &text);
  std::string decode(const std::vector<int64_t> &token_ids);

private:
  std::unordered_map<std::string, int64_t> token_to_id;
  std::unordered_map<int64_t, std::string> id_to_token;
  std::string normalize(const std::string &text);
};
