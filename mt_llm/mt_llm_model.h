
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_MODEL
#define MT_LLM_MODEL

#include "llama.h"

#include "mt_llm_p.h"

/** Returns a vector with one entry for each digit (the numbers from 0 to 9),
 *  where each of these entries is a vector itself holding all token IDs that
 *  may represent that digit (e.g. " 2", "\n2" and "2").
 */
std::vector<std::vector<int>> mt_llm_model_get_digit_tokens(
    struct llama_model const & model);

/**
 * - Returns true, if given model('s name) is supported and given parameters
 *   object was modified.
 *   Returns false, if not supported and nothing was changed.
 */
bool mt_llm_model_try_set_prompts(
    struct llama_model const & model, struct mt_llm_p & p);

/** Initialize model.
 * 
 *  - Caller takes ownership of returned object.
 *  - llama_backend_init() must have already been called. 
 *  - De-initialize model object with llama_free_model(), later.
 */
llama_model* mt_llm_model_create(mt_llm_p const & mt_p);

#endif //MT_LLM_MODEL
