CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPT_FLAGS = -O3 -march=native -ffast-math -static

all: baseline optimized tiled perf

baseline:
	$(CXX) $(CXXFLAGS) -o baseline.exe src/gemm_all.cpp

optimized:
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o optimized.exe src/gemm_all.cpp

tiled:
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o tiled.exe src/gemm_all.cpp

perf:
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o perf.exe src/gemm_all.cpp

clean:
	rm -f baseline.exe optimized.exe tiled.exe perf.exe
