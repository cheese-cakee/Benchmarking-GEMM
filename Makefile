CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPT_FLAGS = -O3 -march=native -ffast-math -static

all: baseline optimized

baseline:
	$(CXX) $(CXXFLAGS) -o baseline.exe gemm_all.cpp

optimized:
	$(CXX) $(CXXFLAGS) $(OPT_FLAGS) -o optimized.exe gemm_all.cpp

clean:
	rm -f baseline.exe optimized.exe
