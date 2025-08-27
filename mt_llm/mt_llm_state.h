
// Marcel Timm, RhinoDevel, 2025feb23

#ifndef MT_LLM_STATE
#define MT_LLM_STATE

#ifdef __cplusplus
    #include <cstdint>
#else //__cplusplus
    #include <stdint.h>
#endif //__cplusplus

struct mt_llm_state
{
    int last_tok_type; // 0
    int tok_cnt; // -1

    uint8_t* state; // nullptr
    size_t size; // 0
};

#endif //MT_LLM_STATE
