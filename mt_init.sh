#!/usr/bin/env bash
set -o nounset
set -o errexit
set -o pipefail

# Marcel Timm, RhinoDevel, 2026jul25

# ******************************************************************************
# *** SYSTEM SETTING (configure for your computer)                           ***
# ******************************************************************************

CUDA=true
JOBS=10

# ******************************************************************************
# *** OPTIONS                                                                ***
# ******************************************************************************

WAIT_FOR_KEY_PRESS_AFTER_RUN=true

# ******************************************************************************
# *** Constants                                                              ***
# ******************************************************************************

LOG_PREFIX=">>> "

# Not really constants, but we act like it:
PATH_ROOT="$PWD"
PATH_LLM="$PATH_ROOT/mt_llm"
PATH_LLAMA="$PATH_LLM/llama.cpp"

# ******************************************************************************
# *** Helper functions                                                       ***
# ******************************************************************************

log() {
    printf '%s%s' "$LOG_PREFIX" "$*"
}

log_line() {
    log "$@"
    printf '\n'
}

wait_for_key_press() {
    log '*** Press any key to continue. ***'
    read -n 1 -s -r </dev/tty
    echo
}

RUN() {
    log_line "EXECUTING: $*"
    "$@"

    if "$WAIT_FOR_KEY_PRESS_AFTER_RUN"; then
        wait_for_key_press
    fi
}

# ******************************************************************************
# *** The actual script ("main")                                             ***
# ******************************************************************************

log_line "System settings:"
log_line "- use CUDA: $CUDA"
log_line "- parallel jobs: $JOBS"
wait_for_key_press

RUN git submodule update --init --recursive

RUN cd $PATH_LLAMA
if "$CUDA"; then
    RUN cmake -B build -DGGML_CUDA=ON
else
    RUN cmake -B build -DGGML_CUDA=OFF
fi
RUN cmake --build build -j "$JOBS" --config Release

RUN cd $PATH_LLM
RUN make