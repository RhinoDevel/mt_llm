# mt_llm

*Marcel Timm, RhinoDevel, 2025*

**mt_llm** is a C++ library for Linux and Windows that offers a pure C interface
to the awesome large-language model inference engine called
[llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov.

**mt_llm** is intended to be used for single-user LLM inference.

**mt_llm** supports:

- Simplified/reduced configuration parameters.
- Simple init./query/reset/deinit. functions.
- Callback to send tokens to and more and let the callback decide, when to stop
  inference.
- Snapshot interface to store/update/reset the current LLM state (using RAM).
- Let the callback retrieve the probabilities of the digits 0 to 9 being the
  next inferred token while ignoring sampling (e.g. for categorization).

## How To

Clone the **mt_llm** repository:

`git clone https://github.com/RhinoDevel/mt_llm.git`

Enter the created folder:

`cd mt_llm`

Get the [llama.cpp](https://github.com/ggml-org/llama.cpp) submodule content:

`git submodule update --init --recursive`

## Linux

No details for Linux here, yet, but you can take a look at the Windows
instructions below and at the [Makefile](./mt_llm/Makefile).

## Windows

#### Note:

All the following examples are building static libraries, there may be use cases
where dynamically linked libraries are sufficient, too.

### Build [llama.cpp](https://github.com/ggml-org/llama.cpp)

#### Compile `llama.lib`, `ggml.lib` and `common.lib` as static libraries

Compile the necessary `llama.lib`, `ggml.lib` and `common.lib` libraries via
Visual Studio and `mt_llm/llama.cpp/CMakeLists.txt` as static libraries.

To do that, select `x64-windows-msvc-debug` or `x64-windows-msvc-release` as
configuration, in Visual Studio.

Also modify the file `mt_llm/llama.cpp/CMakePresets.json` which is created by
Visual Studio:

Add

```
, "BUILD_SHARED_LIBS": "OFF"
, "LLAMA_CURL": "OFF"
```

to the properties of `configurePresets.cacheVariables`, where the `"name"` is
`"base"`.

#### CUDA build

Modify the file `llama.cpp/CMakePresets.json`:

Add

```
, "GGML_CUDA":  "ON"
```

to the properties of `configurePresets.cacheVariable`, where the `"name"` is
`"base"`.

Additionally link **mt_llm** with this (from the llama.cpp build result folder):

```
ggml\src\ggml-cuda\ggml-cuda.lib
```

Additionally link **mt_llm** with these (from the CUDA folder):

```
lib\x64\cublas.lib
lib\x64\cuda.lib
lib\x64\cudart.lib
```

### Test [llama.cpp](https://github.com/ggml-org/llama.cpp) (without mt_llm)

The [llama.cpp](https://github.com/ggml-org/llama.cpp) binaries are also created
by the build described above.

E.g. for a release build, you can find them at
`mt_llm\llama.cpp\build-x64-windows-msvc-release\bin`.

### Build mt_llm

- Open solution `mt_llm.sln` with Visual Studio (tested with 2022).
- Compile in release or debug mode.

### Test mt_llm

- Get the DLL and LIB files resulting from the build, e.g. for release mode
  `x64\Release\mt_llm.dll` and `x64\Release\mt_llm.lib`, copy them to a new
  folder.

- Copy the following header files to that new folder, too:
  - `mt_llm\mt_llm.h`
  - `mt_llm\mt_llm_lib.h`
  - `mt_llm\mt_llm_p.h`
  - `mt_llm\mt_llm_tok_type.h`
  - `mt_llm\mt_llm_snapshot.h`

- Also copy a [supported](mt_llm/mt_llm_model.cpp)
  [GGUF model file](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q5_K_M.gguf?download=true)
  to the same new folder.

- Go to the new folder and create a file `main.c` with the following code:

```
#include "mt_llm.h"
#include "mt_llm_p.h"
#include "mt_llm_tok_type.h"

#include <string.h>
#include <stdio.h>

static bool my_callback(
    int tok,
    char const * piece,
    int type,
    float const * dig_probs)
{
    if(type == MT_TOK_TYPE_SAMPLED_NON_EOG_NON_CONTROL)
    {
        // This example may not display all characters correctly..
        printf("%s", piece);
    }
    else
    {
        if(type == MT_TOK_TYPE_SAMPLED_EOG)
        {
            printf("\n\n");
        }
    }
    return false;
}

int main()
{
    struct mt_llm_p p;
    
    // *****************************
    // *** Setup the parameters: ***
    // *****************************
    
    p.n_gpu_layers = 0;
    
    p.seed = -1;
    p.n_ctx = 2048;
    p.threads = 0;
    p.flash_attn = 0;
    
    p.top_k = 40;
    p.top_p = 0.95;
    p.min_p = 0.05;
    p.temp = 0.8;
    p.grammar[0] = '\0';
    
    strncpy(
        p.model_file_path,
        "gemma-3-1b-it-Q5_K_M.gguf",
        MT_LLM_P_LEN_MODEL_FILE_PATH);

    strncpy(
        p.sys_prompt,
        "You are a helpful AI assistant.",
        MT_LLM_P_LEN_SYS_PROMPT);
    p.prompt_beg_delim[0] = '\0';
    p.prompt_end_delim[0] = '\0';
    p.sys_prompt_beg_delim[0] = '\0';
    p.sys_prompt_mid_delim[0] = '\0';
    p.sys_prompt_end_delim[0] = '\0';
    p.rev_prompt[0] = '\0';
    p.think_beg_delim[0] = '\0';
    p.think_end_delim[0] = '\0';

    p.try_prompts_by_model = true;

    p.callback = my_callback;

    // **************************
    // *** Initialize mt_llm: ***
    // **************************

    mt_llm_reinit(&p); // Ignoring return value, here..
    
    // **********************
    // *** Query the LLM: ***
    // **********************
    
    mt_llm_query("Please tell me a very short story about a dog!");
    
    // (inference is running here, and will call the callback for each token)
    
    // ****************************
    // *** Deinitialize mt_llm: ***
    // ****************************

    mt_llm_deinit();

    return 0;
}
```

- Compile via `cl main.c mt_llm.lib`.
- Run `main.exe`.
