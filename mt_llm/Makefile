
# RhinoDevel, Marcel Timm, 2025feb25

CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17 -fPIC -DNDEBUG

LLAMA_DIR = ./llama.cpp
LLAMA_LIB_DIRS = -L$(LLAMA_DIR)/build/bin -L$(LLAMA_DIR)/build/common

LLAMA_LIBS = -lggml-base -lggml-cpu -lggml -lllama -lcommon
#
# With CUDA:
#
#LLAMA_LIBS = -lggml-base -lggml-cpu -lggml-cuda -lggml -lllama -lcommon

LLAMA_INCLUDES = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/common -I$(LLAMA_DIR)/src

SRC = $(filter-out llama.cpp,$(wildcard *.cpp))
OBJ = $(SRC:.cpp=.o)
LIBRARY = libmtllm.so

$(LIBRARY): $(OBJ)
	$(CXX) -shared -o $@ $^ $(LLAMA_LIB_DIRS) $(LLAMA_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LLAMA_INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(LIBRARY)

.PHONY: clean
