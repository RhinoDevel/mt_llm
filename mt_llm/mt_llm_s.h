
// Marcel Timm, RhinoDevel, 2024aug28

#ifndef MT_LLM_S
#define MT_LLM_S

struct mt_llm_s
{
    int last_tok_type; // 0
    int tok_cnt; // -1
    struct mt_llm_p * mt_p; // nullptr
    struct llama_model * model; // nullptr
    struct llama_context * ctx; // nullptr
    struct llama_sampler * sampler; // nullptr // Better use common_sampler?
};

#endif //MT_LLM_S
