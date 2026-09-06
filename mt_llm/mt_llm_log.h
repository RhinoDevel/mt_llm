
// Marcel Timm, RhinoDevel, 2024aug28

#ifndef MT_LLM_LOG
#define MT_LLM_LOG

#ifdef __cplusplus
    #include <cstdio>
    #include <cstdarg>
#else //__cplusplus
    #include <stdio.h>
    #include <stdarg.h>
    #include <stdbool.h>
#endif //__cplusplus

#define MT_LOG_ERR(fmt, ...) \
    mt_llm_log(stderr, "ERROR: ", __func__, fmt, ##__VA_ARGS__)

#define MT_LOG(fmt, ...) \
    mt_llm_log(stdout, "", __func__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

/**
 * - Do NOT use this function directly, use the macros MT_LOG_ERR and MT_LOG
 *   instead.
 */
void mt_llm_log(
    FILE * const stream,
    char const * prefix,
    char const * func,
    char const * fmt,
    ...);

/** Logging is DISABLED by default. This function has to be called at least once
 *  to enable logging via MT_LOG_ERR and MT_LOG macros.
 */
void mt_llm_log_set_enabled(bool const is_enabled);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //MT_LLM_LOG
