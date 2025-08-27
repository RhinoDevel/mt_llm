
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_P
#define MT_LLM_P

#ifdef __cplusplus
    #include <cstdint>
#else //__cplusplus
    #include <stdint.h>
    #include <stdbool.h>
#endif //__cplusplus

#define MT_LLM_P_LEN_GRAMMAR 511 + 1
#define MT_LLM_P_LEN_MODEL_FILE_PATH 255 + 1
#define MT_LLM_P_LEN_PROMPT_BEG_DELIM 128 + 1
#define MT_LLM_P_LEN_PROMPT_END_DELIM 128 + 1
#define MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM 128 + 1
#define MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM 128 + 1
#define MT_LLM_P_LEN_SYS_PROMPT_END_DELIM 128 + 1
#define MT_LLM_P_LEN_SYS_PROMPT 511 + 1
#define MT_LLM_P_LEN_REV_PROMPT 64 + 1
#define MT_LLM_P_LEN_THINK_BEG_DELIM 64 + 1
#define MT_LLM_P_LEN_THINK_END_DELIM 64 + 1

/**
 * - This should/must be compatible with (pure-)C. 
 */
struct mt_llm_p
{
    // **************************
    // *** llama_model_params ***
    // **************************

    int32_t n_gpu_layers;

    // ****************************
    // *** llama_context_params ***
    // ****************************

    uint32_t seed; // -1 = Random number.
    uint32_t n_ctx; // 0 = Use context size of model.
    uint32_t threads; // Number of threads to use for inference
                      // (0 = system-dependent).

    uint8_t flash_attn; // 0 = false, otherwise true.

    // *****************************
    // *** llama_sampling_params ***
    // *****************************
    
    int32_t top_k; // top_k <= 0 to use vocabulary's size.
    float top_p; // 1.0f to disable.
    float min_p; // 0.0f to disable.
    float temp;  // temp <= 0.0 to sample greedily, 0.0 to not output probabil.
    char grammar[MT_LLM_P_LEN_GRAMMAR]; // Optional BNF-like grammar to
                                        // constrain sampling.

    // **************
    // *** Others ***
    // **************

    char model_file_path[MT_LLM_P_LEN_MODEL_FILE_PATH]; 

    char sys_prompt[MT_LLM_P_LEN_SYS_PROMPT]; // E.g.: "You are a helpful AI."
    char prompt_beg_delim[MT_LLM_P_LEN_PROMPT_BEG_DELIM];
    char prompt_end_delim[MT_LLM_P_LEN_PROMPT_END_DELIM];
    char sys_prompt_beg_delim[MT_LLM_P_LEN_SYS_PROMPT_BEG_DELIM];
    char sys_prompt_mid_delim[MT_LLM_P_LEN_SYS_PROMPT_MID_DELIM];
    char sys_prompt_end_delim[MT_LLM_P_LEN_SYS_PROMPT_END_DELIM];
    char rev_prompt[MT_LLM_P_LEN_REV_PROMPT]; // E.g.: "Master:"
    char think_beg_delim[MT_LLM_P_LEN_THINK_BEG_DELIM];
    char think_end_delim[MT_LLM_P_LEN_THINK_END_DELIM];

    // If set to "true", default settings based on the model's name (as found in
    // its meta data) will be used for the ...prompt... properties' values,
    // if such defaults exist for the model (but not sys_prompt): 
    //
    uint8_t try_prompts_by_model; // 0 = false, true otherwise.

    // Retrieves each token, the token's string representation and under some
    // circumstances the probabilities for the digits (0 to 9), which will be
    // NULL, if not given:
    //
    bool(*callback)(int, char const *, int, float const *);
};

// The following functions are used internally and are not pure-C:
//
#ifdef __cplusplus/*MT_EXPORT_LLM*/

void mt_llm_p_print(struct mt_llm_p const & mt_p);

/**
 * - Only use this on an object created via mt_llm_p_create_copy()!
 * - Also frees property values, where necessary.
 */
void mt_llm_p_free(struct mt_llm_p * const mt_p);

/** Deep-copies given object.
 * 
 * - Use mt_llm_p_free() to delete object created with this function.
 * - Caller takes ownership of returned object (and its property values, if
 *   necessary).
 */
struct mt_llm_p * mt_llm_p_create_copy(struct mt_llm_p const & mt_p);

#endif //__cplusplus/*MT_EXPORT_LLM*/

#endif //MT_LLM_P
