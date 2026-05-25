#include<iostream>
#include<cstdlib>
#include<chrono>
#include<immintrin.h>

namespace workshop{

   template<typename T>
   struct Array {
      T* data_;
      size_t size_;
      Array(size_t n) : size_(n) {data_ = new T[n]; }
      ~Array() { delete[] data_; }
      T& operator[](size_t i) { return data_[i]; }
      const T& operator[](size_t i) const{ return data_[i]; }
   };

   template<typename T>
   struct AlignedArray {
      T*data_;
      size_t size_;
      AlignedArray(size_t n) : size_(n) {
         data_ = (T*)_mm_malloc(n * sizeof(T), 32);
      }
      ~AlignedArray(){ _mm_free(data_);}
      T& operator[](size_t i){ return data_[i]; }
      const T& operator[](size_t i) const { return data_[i]; }
   };

   using Clock = std::chrono::high_resolution_clock;
   
   inline auto start_timer() { return Clock::now(); }

   template<typename T>
   inline double get_duration(T start){
      auto end = Clock::now();
      return std::chrono::duration<double, std::milli>(end - start).count();
   }
}

__attribute__((optimize("-fno-tree-vectorize")))
static void scalar_add(float* c, const float* a, const float* b, int size, int iterations) {
    for (int j = 0; j < iterations; ++j)
        for (int i = 0; i < size; ++i)
            c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
   const int size = 512;

   auto a = workshop::Array<float>(size);
   auto b = workshop::Array<float>(size);
   auto c = workshop::Array<float>(size);
   
   
   auto avx_a = workshop::AlignedArray<__m256>(size/8);
   auto avx_b = workshop::AlignedArray<__m256>(size/8);
   auto avx_c = workshop::AlignedArray<__m256>(size/8);

   for(int i =0;i<size;++i){
      a[i] = 1.0*(i+1);
      b[i] = 2.5*(i+1);
      c[i] = 0.0;
   }

   for(int i =0;i<size;i += 8)
   {
        avx_a[i/8] = _mm256_set_ps(1.0*(i+7+1),
                                   1.0*(i+6+1),
                                   1.0*(i+5+1),
                                   1.0*(i+4+1),
                                   1.0*(i+3+1),
                                   1.0*(i+2+1),
                                   1.0*(i+1+1),
                                   1.0*(i+0+1));

        avx_b[i/8] = _mm256_set_ps(2.5*(i+7+1),
                                   2.5*(i+6+1),
                                   2.5*(i+5+1),
                                   2.5*(i+4+1),
                                   2.5*(i+3+1),
                                   2.5*(i+2+1),
                                   2.5*(i+1+1),
                                   2.5*(i+0+1));

       avx_c[i/8] = _mm256_set1_ps(0.0);
   }
   
auto timer = workshop::start_timer();
scalar_add(c.data_, a.data_, b.data_, size, 100000);
auto duration = workshop::get_duration(timer);

   timer = workshop::start_timer();

   for(int j =0;j<100000;++j)
   {
      for(int i =0;i<size/8;++i)
      {
         avx_c[i] = _mm256_add_ps(avx_a[i],avx_b[i]);
      }
   }

   auto vector_duration = workshop::get_duration(timer);

   std::cout << "The standard loop took " << duration
             << " ms to complete."<< std::endl;
   std::cout << "The vectorised loop took " << vector_duration
             << " ms to complete." << std::endl;
}