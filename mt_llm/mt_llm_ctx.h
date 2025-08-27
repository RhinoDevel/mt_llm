
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_CTX
#define MT_LLM_CTX

#include "sampling.h"
#include "llama.h"

#include "mt_llm_p.h"

std::vector<int> mt_llm_ctx_tokenize(
    llama_context const & ctx, char const * const str, bool const add_special);

std::string mt_llm_ctx_get_piece_from(
    llama_context& ctx, llama_token const tok);

/** Add given tokens to the context. Inform sampler about the new tokens. Call
 *  callback.
 * 
 * - Uses 1 as "batch" size.
 * - Never applies grammar.
 */
bool mt_llm_ctx_decode(
    llama_context& ctx,
    llama_sampler& sampling_ctx,
    int const existing_token_count,
    std::vector<int> const & tokens,
    bool(*callback)(llama_token, std::string const &, std::vector<float> const &));

/**
 * - Prepends BOS token, if existing token count is zero and model meta data
 *   says so.
 * - Returns the count of tokens for string given or negative value, if an error
 *   occurred.
 * - Uses batch size of 1.
 * - Never applies grammar.
 */
int mt_llm_ctx_decode(
        llama_context& ctx,
        llama_sampler& sampling_ctx,
        int const existing_token_count,
        char const * const str,
        bool(*callback)(llama_token, std::string const &, std::vector<float> const &));

/** Initialize the model.
 * 
 *  - Caller takes ownership of created object.
 *  - Use llama_free() to de-initialize.
 */
llama_context* mt_llm_ctx_create(
    mt_llm_p const & mt_p, llama_model& model);

#endif //MT_LLM_CTX
