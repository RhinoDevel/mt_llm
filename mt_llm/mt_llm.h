
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

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

MT_EXPORT_LLM_API void __stdcall mt_llm_free(void * const ptr);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Returns -1, if not initialized.
 * - Returns -2, if NULL given.
 * - Returns -3, if given slot index is invalid.
 * - Returns -4, if configured to create embeddings or for reranking usage.
 */
MT_EXPORT_LLM_API int __stdcall mt_llm_get_token_count(
    char const * const text, bool const add_special, int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Caller takes ownership of return value.
 *   mt_llm_state.state and the mt_llm_state object itself can be freed via
 *   mt_llm_free() each.
 * 
 * - Returns NULL and does nothing, if invalid slot index given.
 * - Returns NULL and does nothing, if not initialized.
 * - Returns NULL, if dumping to memory failed (e.g. out of space).
 */
MT_EXPORT_LLM_API struct mt_llm_state * __stdcall mt_llm_state_create(
    int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Returns false and does nothing, if invalid slot index given.
 * - Returns false and does nothing, if not initialized.
 * 
 * - Assumes non-NULL given and object to hold valid values.
 * 
 * - If false is returned because of failed read attempt, the state of the
 *   context is unknown..
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_state_restore(
    struct mt_llm_state const * const state, int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Returns false and does nothing, if not initialized or invalid slot index or
 *   configured for embeddings creation.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_query(
    char const * const prompt,
    int const slot_index,
    bool const skip_sys_prompt_end_delim_and_inference,
    bool const follow_up_decode_prompt_and_sys_prompt_end_delim);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Caller takes ownership of the returned array, must be freed via
 *   mt_llm_free().
 * 
 * - Returns NULL and does nothing, if not initialized or invalid slot index
 *   or NOT configured for embeddings creation.
 * 
 * - Clears the memory (K/V) before creation of embeddings.
 * 
 * - The returned embedding vector is (Euclidean) normalized (pooling type used
 *   is "mean").
 */
MT_EXPORT_LLM_API float* __stdcall mt_llm_create_embeddings(
    char const * const prompt, int const slot_index, int * const out_count);

/**
 * - Slot index may be 0 or 1.
 *
 * - Caller takes ownership of the returned array, must be freed via
 *   mt_llm_free().
 *
 * - Returns NULL and does nothing, if not initialized or invalid slot index
 *   or NOT configured for reranker "mode" or some other error happened.
 *
 * - Clears the memory (K/V) before creation of reranking result.
 *
 * - The returned numbers are NOT normalized, yet.
 * 
 * - The returned numbers are of the same count as the given documents.
 */
MT_EXPORT_LLM_API float* __stdcall mt_llm_rerank(
    char const * const query,
    char const * const * documents,
    int const doc_count,
    int const slot_index);

/** Reset state, as if the model just got loaded. Optionally update system
 *  prompt.
 *
 * - Slot index may be 0 or 1.
 * 
 * - Does nothing, if singleton or slot is not initialized.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_reset(
    char const * const sys_prompt, int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * 
 * - Does no harm, if not initialized.
 * 
 * - At the end, call this once for each slot used.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_deinit(int const slot_index);

/**
 * - Slot index may be 0 or 1.
 * 
 * - De-initializes first, if already initialized.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_reinit(
    struct mt_llm_p const * const mt_p, int const slot_index);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM
