
// Marcel Timm, RhinoDevel, 2025feb23

// This is meant to be a pure-C interface to run LLM inference.

#ifndef MT_LLM_SNAPSHOT
#define MT_LLM_SNAPSHOT

#include "mt_llm_lib.h"

#ifdef __cplusplus
	#include <cstdbool>
#else //__cplusplus
	#include <stdbool.h>
#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

MT_EXPORT_LLM_API void __stdcall mt_llm_snapshot_clear();
MT_EXPORT_LLM_API bool __stdcall mt_llm_snapshot_update();
MT_EXPORT_LLM_API bool __stdcall mt_llm_snapshot_restore();

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM_SNAPSHOT
