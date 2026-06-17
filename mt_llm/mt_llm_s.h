
// Marcel Timm, RhinoDevel, 2024aug28

#ifndef MT_LLM_S
#define MT_LLM_S

#include "mt_llm_model.h"

struct mt_llm_s
{
    int last_tok_type; // 0
    struct mt_llm_p * mt_p; // nullptr
    struct mt_llm_model * model; // nullptr
    struct llama_context * ctx; // nullptr
    struct llama_sampler * sampler; // nullptr
};

#endif //MT_LLM_S
