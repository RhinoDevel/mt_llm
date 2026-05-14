
// Marcel Timm, RhinoDevel, 2026may13

// This is meant to be a pure-C interface to run LLM inference.

#ifndef MT_LLM_IMPORT
#define MT_LLM_IMPORT

#include "mt_llm_lib.h"

#ifdef __cplusplus
	#include <cstdbool>
#else //__cplusplus
	#include <stdbool.h>
#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

/**
 *  - First message given will always be interpreted as the system prompt,
 *    second message as first user prompt, third message as first LLM response
 *    and so on.
 *  - Always starts import at the beginning of the context.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_import(
	char const * const * const messages,
	int const msg_cnt,
	int const slot_index);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM_IMPORT
