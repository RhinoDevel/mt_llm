
// Marcel Timm, RhinoDevel, 2026sep04

#include <cstdio>
#include <cstdarg>
#include <cassert>

#include "mt_llm_log.h"

static bool s_log_is_enabled = true; // Enabled by default.

void mt_llm_log(
    FILE * const stream,
    char const * prefix,
    char const * func,
    char const * fmt,
    ...)
{
    assert(stream != nullptr);
    assert(prefix != nullptr);
    assert(fmt != nullptr);
    assert(func != nullptr);

    va_list args;

    if(!s_log_is_enabled)
    {
        return;
    }

    fprintf(stream, "%s : %s", func, prefix);

    va_start(args, fmt);
    vfprintf(stream, fmt, args);
    va_end(args);
}

void mt_llm_log_set_enabled(bool const is_enabled)
{
    s_log_is_enabled = is_enabled;
}