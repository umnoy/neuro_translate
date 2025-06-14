// Отключение аннотаций SAL
#define _SAL_VERSION 20
#define _Out_
#define _In_
#define _In_opt_
#define _Frees_ptr_opt_
#define _Check_return_
#define _Success_(expr)
#define _Ret_maybenull_
#define _Null_terminated_
#define _Outptr_opt_result_maybenull_


#include "../tokenizer/tokenizer.hpp"
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace std;

class Translator {
public:
  Translator(const Tokenizer tokenizer, const string &encoder_path,
             const string &decoder_path, int pad_token_id, int eos_token_id,
             int max_length = 50, int beam_width = 3);

  string run(const string &input);

private:
  Ort::Env env;
  Ort::Session encoder_session;
  Ort::Session decoder_session;
  Ort::SessionOptions session_options;
  Ort::AllocatorWithDefaultOptions allocator;

  int pad_token_id;
  int eos_token_id;
  int max_length;
  int beam_width;

  Tokenizer tokenizer;

  vector<float> encode_input(const vector<int64_t> &input_ids);
  vector<float> decode_step(const vector<int64_t> &input_ids,
                            const vector<int64_t> &encoder_input_ids,
                            const vector<float> &encoder_hidden_state);
  string decode_ids(const vector<int64_t> &ids);

  vector<float> softmax(const vector<float> &logits);
  vector<pair<int64_t, float>> top_k(const vector<float> &probs, int k);
};

struct Beam {
  vector<int64_t> tokens;
  float score;

  bool operator<(const Beam &other) const { return score < other.score; }
};
