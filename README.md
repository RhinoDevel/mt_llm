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

...