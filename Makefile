CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPT_FLAGS = -O3 -march=native -ffast-math -fopenmp
# Unoptimized build (shows compiler's baseline)
debug:
	$(CXX) $(CXXFLAGS) -O0 -o gemm_debug.exe src/gemm_all.cpp
# Fully optimized build
release:
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o gemm_bench.exe src/gemm_all.cpp
all: release
clean:
	rm -f gemm_debug.exe gemm_bench.exe