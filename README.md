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

...

### Test [llama.cpp](https://github.com/ggml-org/llama.cpp) (without mt_llm)

...

### Build mt_llm

- Open solution `mt_llm.sln` with Visual Studio (tested with 2022).
- Compile in release or debug mode.

### Test mt_llm

...