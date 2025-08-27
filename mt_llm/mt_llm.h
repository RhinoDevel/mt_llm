
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

/**
 * - Returns -1, if not initialized.
 * - Returns -2, if nullptr given.
 */
MT_EXPORT_LLM_API int __stdcall mt_llm_get_token_count(
    char const * const text, bool const add_special);

/**
 * - Caller takes ownership of return value.
 * - Returns nullptr and does nothing, if not initialized.
 * - Returns nullptr, if dumping to memory failed (e.g. out of space).
 */
MT_EXPORT_LLM_API struct mt_llm_state * __stdcall mt_llm_state_create();

/**
 * - Returns false and does nothing, if not initialized.
 * - Assumes non-nullptr given and object to hold valid values.
 * - If false is returned because of failed read attempt, the state of the
 *   context is unknown..
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_state_restore(
    struct mt_llm_state const * const state);

/**
 * - Returns false and does nothing, if not initialized.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_query(char const * const prompt);

/** Reset state, as if the model just got loaded.
 *
 * - Does nothing, if singleton is not initialized.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_reset();

/**
 * - Does no harm, if not initialized.
 */
MT_EXPORT_LLM_API void __stdcall mt_llm_deinit();

/**
 * - De-initializes first, if already initialized.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_reinit(
    struct mt_llm_p const * const mt_p);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM
