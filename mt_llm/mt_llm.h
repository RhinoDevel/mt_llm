
// Marcel Timm, RhinoDevel, 2024aug21

// This is meant to be a pure-C interface to run LLM inference.

#ifndef MT_LLM
#define MT_LLM

#include "mt_llm_lib.h"

#ifdef __cplusplus
    #include <cstdbool>
    #include <cstdint>
#else //__cplusplus
    #include <stdbool.h>
    #include <stdint.h>
#endif //__cplusplus

#include "mt_llm_p.h"

#define MT_LLM_STR_LIM 65535

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

MT_EXPORT_LLM_API void __stdcall mt_llm_free(void * const ptr);

/** Return maximum token count (context length) actually used (as the value is
 *  unknown to the user, if mt_llm_p.n_ctx is set to 0).
 * 
 * - Slot index may be 0 or 1.
 * - Returns -1, if not initialized.
 * - Returns -2, if given slot index is invalid.
 */
MT_EXPORT_LLM_API int __stdcall mt_llm_get_max_token_count(
    int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * - Returns -1, if not initialized.
 * - Returns -2, if NULL given.
 * - Returns -3, if given slot index is invalid.
 * - Returns -4, if configured to create embeddings or for reranking usage.
 */
MT_EXPORT_LLM_API int __stdcall mt_llm_get_token_count(
    char const * const text, bool const add_special, int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * - Caller takes ownership of return value.
 *   mt_llm_state.state and the mt_llm_state object itself can be freed via
 *   mt_llm_free() each.
 * - Returns NULL and does nothing, if invalid slot index given.
 * - Returns NULL and does nothing, if not initialized.
 * - Returns NULL, if dumping to memory failed (e.g. out of space).
 */
MT_EXPORT_LLM_API struct mt_llm_state * __stdcall mt_llm_state_create(
    int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * - Returns false and does nothing, if invalid slot index given.
 * - Returns false and does nothing, if not initialized.
 * - Assumes non-NULL given and object to hold valid values.
 * - If false is returned because of failed read attempt, the state of the
 *   context is unknown..
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_state_restore(
    struct mt_llm_state const * const state, int const slot_index);

/** Add ("decode") given text as system prompt to the context.
 *
 *  - Slot index may be 0 or 1.
 *  - Automatically inserts prompt_template.sys_prompt_beg_delim and
 *    .sys_prompt_mid_delim around the given system prompt, too.
 *  - Totally ignores slot's configured system prompt.
 *  - Returns false and does nothing, if not initialized or invalid slot index
 *    or configured for embeddings creation or reranking or configured to use a
 *    model in thinking/reasoning mode or token count of context is not zero.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_decode_sys_prompt(
    char const * const sys_prompt, int const slot_index);

/** Add ("decode") given text to the context, as if it were a query/request by
 *  the user.
 *
 *  - Makes sense directly after a prompt_template.sys_prompt_mid_delim or
 *    .prompt_beg_delim (see mt_lm_model.cpp), only (which is not checked,
 *    here).
 *  - Slot index may be 0 or 1.
 *  - Automatically inserts prompt_template.sys_prompt_end_delim or
 *    .prompt_end_delim (depending on bool input parameter) after request, too.
 *  - Returns false and does nothing, if not initialized or invalid slot index
 *    or configured for embeddings creation or reranking or configured to use a
 *    model in thinking/reasoning mode.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_decode_request(
    char const * const request, int const slot_index, bool const is_first);

/** Add ("decode") given text to the context, as if it were a response generated
 *  by the model.
 * 
 *  - Makes sense directly after a prompt_template.prompt_end_delim (see
 *    mt_lm_model.cpp), only (which is not checked, here).
 *  - Automatically inserts EOG token and prompt_template.prompt_beg_delim after
 *    response, too.
 *  - Slot index may be 0 or 1.
 *  - Returns false and does nothing, if not initialized or invalid slot index
 *    or configured for embeddings creation or reranking or configured to use a
 *    model in thinking/reasoning mode.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_decode_response(
    char const * const response,
    int const slot_index,
    bool const no_last_prompt_beg_delim);

/**
 * - Slot index may be 0 or 1.
 * - Returns false and does nothing, if not initialized or invalid slot index or
 *   configured for embeddings creation or reranking.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_query(
    char const * const prompt,
    int const slot_index,
    bool const skip_sys_prompt_end_delim_and_inference,
    bool const follow_up_decode_prompt_and_sys_prompt_end_delim);

/**
 * - Slot index may be 0 or 1.
 * - Caller takes ownership of the returned array, must be freed via
 *   mt_llm_free().
 * - Returns NULL and does nothing, if not initialized or invalid slot index
 *   or NOT configured for embeddings creation.
 * - Clears the memory (K/V) before creation of embeddings.
 * - The returned embedding vector is (Euclidean) normalized (pooling type used
 *   is "mean").
 */
MT_EXPORT_LLM_API float* __stdcall mt_llm_create_embeddings(
    char const * const prompt, int const slot_index, int * const out_count);

/**
 * - Slot index may be 0 or 1.
 * - Caller takes ownership of the returned array, must be freed via
 *   mt_llm_free().
 * - Returns NULL and does nothing, if not initialized or invalid slot index
 *   or NOT configured for reranker "mode" or some other error happened.
 * - Clears the memory (K/V) before creation of reranking result.
 * - The returned numbers are NOT normalized, yet.
 * - The returned numbers are of the same count as the given documents.
 */
MT_EXPORT_LLM_API float* __stdcall mt_llm_rerank(
    char const * const query,
    char const * const * const documents,
    int const doc_count,
    int const slot_index);

/** Reset state, as if the model just got loaded. Optionally update system
 *  prompt.
 *
 * - Slot index may be 0 or 1.
 * - Does nothing, if singleton or slot is not initialized.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_reset(
    char const * const sys_prompt, int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * - Does no harm, if not initialized.
 * - At the end, call this once for each slot used.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_deinit(int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * - Only calls mt_llm_reset() instead of full re-initialization, if possible.
 * - De-initializes first, if already initialized.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_reinit(
    struct mt_llm_p const * const mt_p, int const slot_index);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM
