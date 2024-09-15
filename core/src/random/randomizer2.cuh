//#pragma once
//
//#include <curand.h>
//#include <curand_kernel.h>
//#include <cuda_runtime.h>
//#include <iostream>
//
//struct CurandRNG
//{
//    bool use_gpu;  // Flag to determine if we are using GPU or CPU
//    size_t n = 1; // Default number of random numbers to generate
//
//    curandGenerator_t generator;   // cuRAND generator for GPU
//    curandGenerator_t host_generator; // cuRAND generator for CPU
//
//    // Initialize the generator based on whether we want GPU or CPU RNG
//    void init(bool gpu) {
//        use_gpu = gpu;
//        if (use_gpu) {
//            // Initialize cuRAND for GPU
//            curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//            curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
//            printf("Initialized GPU-based random number generator using cuRAND.\n");
//        }
//        else {
//            // Initialize cuRAND for CPU
//            curandCreateGeneratorHost(&host_generator, CURAND_RNG_PSEUDO_DEFAULT);
//            curandSetPseudoRandomGeneratorSeed(host_generator, time(NULL));
//            printf("Initialized CPU-based random number generator using cuRAND.\n");
//        }
//    }
//
//    // Cleanup cuRAND resources
//    void cleanup() {
//        if (use_gpu) {
//            curandDestroyGenerator(generator);
//        }
//        else {
//            curandDestroyGenerator(host_generator);
//        }
//    }
//
//    // Generate random floats between min and max
//    void generate_random_float(float min_val, float max_val) {
//        float* data = new float[n];
//
//        if (use_gpu) {
//            // GPU: Allocate memory on the device
//            float* d_data;
//            cudaMalloc((void**)&d_data, n * sizeof(float));
//
//            // Generate random floats between 0 and 1 on the GPU
//            curandGenerateUniform(generator, d_data, n);
//
//            // Copy the results to the host
//            cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
//
//            // Scale the random values to [min_val, max_val]
//            for (size_t i = 0; i < n; ++i) {
//                data[i] = min_val + data[i] * (max_val - min_val);
//            }
//
//            // Free GPU memory
//            cudaFree(d_data);
//        }
//        else {
//            // CPU: Generate random floats between 0 and 1 on the CPU
//            curandGenerateUniform(host_generator, data, n);
//
//            // Scale the random values to [min_val, max_val]
//            for (size_t i = 0; i < n; ++i) {
//                data[i] = min_val + data[i] * (max_val - min_val);
//            }
//        }
//
//        // Print the results
//        //print_data(data, n, "Random floats between " + std::to_string(min_val) + " and " + std::to_string(max_val));
//
//        // Free host memory
//        delete[] data;
//    }
//
//    // Generate random integers between min and max
//    void generate_random_int(int min_val, int max_val)
//    {
//        float* data = new float[n];  // We'll use floats and cast them to integers after scaling
//
//        if (use_gpu) {
//            // GPU: Allocate memory on the device
//            float* d_data;
//            cudaMalloc((void**)&d_data, n * sizeof(float));
//
//            // Generate random floats between 0 and 1 on the GPU
//            curandGenerateUniform(generator, d_data, n);
//
//            // Copy the results to the host
//            cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
//
//            // Scale and convert to integers
//            for (size_t i = 0; i < n; ++i) {
//                data[i] = min_val + static_cast<int>(data[i] * (max_val - min_val + 1));
//            }
//
//            // Free GPU memory
//            cudaFree(d_data);
//        }
//        else {
//            // CPU: Generate random floats between 0 and 1 on the CPU
//            curandGenerateUniform(host_generator, data, n);
//
//            // Scale and convert to integers
//            for (size_t i = 0; i < n; ++i) {
//                data[i] = min_val + static_cast<int>(data[i] * (max_val - min_val + 1));
//            }
//        }
//
//        // Print the results as integers
//        print_data(reinterpret_cast<int*>(data), n, "Random integers between ");
//
//        // Free host memory
//        delete[] data;
//    }
//
//private:
//    // Print the generated random numbers (for floats and integers)
//    template <typename T>
//    void print_data(const T* data, size_t n, const std::string& label) {
//        std::cout << label << ":" << std::endl;
//        for (size_t i = 0; i < n; ++i) {
//            printf("Random ::: %i", data[i]);
//        }
//        std::cout << std::endl;
//    }
//};
//
////int main() {
////    // User choice for GPU or CPU random number generation
////    bool use_gpu;
////    std::cout << "Enter 1 to use GPU-based RNG, 0 for CPU-based RNG: ";
////    std::cin >> use_gpu;
////
////    CurandRNG rng;
////    rng.init(use_gpu);
////
////    // Generate random floats between 5.0 and 10.0
////    rng.generate_random_float(5.0f, 10.0f);
////
////    // Generate random integers between 1 and 100
////    rng.generate_random_int(1, 100);
////
////    rng.cleanup();
////    return 0;
////}
