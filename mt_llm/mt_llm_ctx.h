
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_CTX
#define MT_LLM_CTX

#include "sampling.h"
#include "llama.h"

#include "mt_llm_p.h"
#include "mt_llm_model.h"

std::vector<llama_token> mt_llm_ctx_tokenize(
    llama_context const & ctx, char const * const str, bool const add_special);

std::string mt_llm_ctx_get_piece_from(
    llama_context& ctx, llama_token const tok);

/**
 * - Prepends BOS token, if existing token count is zero and model meta data
 *   says so.
 * - Decoded tokens counter will hold correct value on error, too.
 * - Uses batch size of 1.
 * - Never applies grammar.
 */
bool mt_llm_ctx_decode(
    llama_context& ctx,
    llama_sampler& sampling_ctx,
    int const existing_token_count,
    char const * const str,
    int& decoded_token_count,
    bool(*callback)(llama_token, std::string const &, std::vector<float> const &));

/** Initialize the model.
 * 
 *  - Caller takes ownership of created object.
 *  - Use llama_free() to de-initialize.
 */
llama_context* mt_llm_ctx_create(
    mt_llm_p const & mt_p, mt_llm_model const & model);

#endif //MT_LLM_CTX
