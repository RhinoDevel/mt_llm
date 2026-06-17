
// Marcel Timm, RhinoDevel, 2024aug21

#ifndef MT_LLM_CTX
#define MT_LLM_CTX

#include "sampling.h"
#include "llama.h"

#include "mt_llm_p.h"
#include "mt_llm_model.h"
#include "mt_llm_s.h"

std::vector<llama_token> mt_llm_ctx_tokenize(
    llama_context const & ctx, char const * const str, bool const add_special);

int mt_llm_ctx_get_tok_cnt(llama_context const & ctx);

std::string mt_llm_ctx_get_piece_from(
    llama_context& ctx, llama_token const tok);

/** Add given tokens to the context. Inform sampler about the new tokens. Call
 *  callback (if given).
 *
 * - Prepends BOS token, if existing token count is zero and model meta data
 *   says so. Given tokens vector will be AUGMENTED in this case!
 * - Decoded tokens counter will hold correct value on error, too.
 * - Never applies grammar.
 */
bool mt_llm_ctx_decode(
    mt_llm_s const &s,
    std::vector<llama_token>& tokens,
    int& decoded_token_count,
    bool(*callback)(llama_token, std::string const &, std::vector<float> const &, mt_llm_s const &),
    std::vector<float> const & dig_probs,
    bool& irq);

/** Initialize the model.
 * 
 *  - Caller takes ownership of created object.
 *  - Use llama_free() to de-initialize.
 */
llama_context* mt_llm_ctx_create(
    mt_llm_p const & mt_p, mt_llm_model const & model);

#endif //MT_LLM_CTX
