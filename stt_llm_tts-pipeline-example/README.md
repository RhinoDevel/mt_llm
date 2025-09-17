# STT -> LLM -> TTS pipeline example

*Marcel Timm, RhinoDevel, 2025*

An example about how to implement a

**S**peech-**T**o-**T**ext -> **L**arge-**L**anguage-**M**odel -> **T**ext-**T**o-**S**peech

pipeline in C, via mt_stt, mt_llm and mt_tts, which are C/C++ wrapper
libraries for Whisper.cpp, Llama.cpp and Piper.

## Note

The instructions are for Windows, but the code should work in Linux, too.

Take a look at the makefiles of
[mt_stt](https://github.com/RhinoDevel/mt_stt/blob/main/mt_stt/Makefile),
[mt_tts](https://github.com/RhinoDevel/mt_tts/blob/master/mt_tts/Makefile)
and [mt_llm](../mt_llm/Makefile) for hints.

## How To

- Follow the instructions to test TTS via
  [mt_tts](https://github.com/RhinoDevel/mt_tts?tab=readme-ov-file#test-mt_tts) first.

- Then update the example project by adding STT as described for
  [mt_stt](https://github.com/RhinoDevel/mt_stt?tab=readme-ov-file#test-mt_stt).

- Proceed by creating the LLM example with the instructions for
  [mt_llm](https://github.com/RhinoDevel/mt_llm?tab=readme-ov-file#test-mt_llm)
  by using **the same folder**.

- Replace `main.c` with the file found [here](./main.c).

- Open `x64 Native Tools Command Prompt for VS 2022` commandline.

- Compile via `cl main.c mt_stt.lib mt_llm.lib mt_tts.lib`.

- Run `main.exe`.

- Listen to the wave file created.