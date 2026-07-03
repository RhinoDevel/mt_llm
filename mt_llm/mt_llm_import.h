
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
 *  - If message count is odd, first message given will be interpreted as the
 *    system prompt, the second message as first user prompt, the third message
 *    as first LLM response and so on.
 *  - If message count is even, first message given will be interpreted as first
 *    user prompt to import, the second message as LLM response and so on.
 *  - Always starts import at the beginning of the context.
 */
MT_EXPORT_LLM_API bool __stdcall mt_llm_import(
	char const * const * const messages,
	int const msg_cnt,
	int const slot_index,
	bool const no_last_prompt_beg_delim);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM_IMPORT
