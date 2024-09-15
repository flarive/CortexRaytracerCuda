//#pragma once
//
//#include <curand.h>
//
//// Structure to manage both CPU and GPU generators
//struct RNG
//{
//    curandGenerator_t generator;  // CURAND generator
//    bool use_gpu;                 // Flag to determine if we are using GPU or CPU
//    int n = 1;
//
//
//    // Initialize the generator based on whether we want GPU or CPU RNG
//    void init(bool gpu)
//    {
//        use_gpu = gpu;
//        if (use_gpu)
//        {
//            // Create GPU-based generator
//            curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//            curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
//            printf("Initialized GPU-based random number generator.\n");
//        }
//        else
//        {
//            // Create CPU-based generator
//            curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
//            curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
//            printf("Initialized CPU-based random number generator.\n");
//        }
//    }
//
//    // Generate random numbers on either the GPU or CPU
//    void generate_random_float(float* result)
//    {
//        if (use_gpu) {
//            // GPU-based random number generation (on device memory)
//            curandGenerateUniform(generator, result, 1);
//        }
//        else {
//            // CPU-based random number generation (on host memory)
//            curandGenerateUniform(generator, result, 1);
//        }
//    }
//
//    // Generate random integers between min and max
//    void generate_random_int(int* result, int min_val, int max_val)
//    {
//        float* data = new float[n];  // We'll use floats and cast them to integers after scaling
//
//        if (use_gpu)
//        {
//            // GPU: Allocate memory on the device
//            // Generate random floats between 0 and 1 on the GPU
//            curandGenerateUniform(generator, data, n);
//
//            // Scale and convert to integers
//            for (size_t i = 0; i < n; ++i) {
//                *result = min_val + static_cast<int>(data[i] * (max_val - min_val + 1));
//            }
//        }
//        else {
//            // CPU: Generate random floats between 0 and 1 on the CPU
//            curandGenerateUniform(generator, data, n);
//
//            // Scale and convert to integers
//            for (size_t i = 0; i < n; ++i) {
//                *result = min_val + static_cast<int>(data[i] * (max_val - min_val + 1));
//            }
//        }
//    }
//
//
//
//
//    // Generate random numbers on either the GPU or CPU
//    void generate_random_int(unsigned int* result)
//    {
//        if (use_gpu) {
//            // GPU-based random number generation (on device memory)
//            curandGenerate(generator, result, 1);
//        }
//        else {
//            // CPU-based random number generation (on host memory)
//            curandGenerate(generator, result, 1);
//        }
//    }
//
//
//
//    // Clean up
//    void cleanup()
//    {
//        curandDestroyGenerator(generator);
//        printf("Cleaned up the random number generator.\n");
//    }
//
//private:
//    // Print the generated random numbers (for floats and integers)
//    template <typename T>
//    void print_data(const T* data, size_t n, const std::string& label) {
//        std::cout << label << ":" << std::endl;
//        for (size_t i = 0; i < n; ++i) {
//            std::cout << data[i] << " ";
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
////    size_t n = 10;  // Number of random numbers to generate
////    float* data;
////
////    RNG rng;
////    rng.init(use_gpu);
////
////    // Allocate memory based on the target (GPU or CPU)
////    if (use_gpu) {
////        // GPU allocation
////        cudaMalloc(&data, n * sizeof(float));
////    }
////    else {
////        // CPU allocation
////        data = (float*)malloc(n * sizeof(float));
////    }
////
////    // Generate random numbers
////    rng.generate(data, n);
////
////    // Copy data from GPU if needed
////    if (use_gpu) {
////        float* host_data = (float*)malloc(n * sizeof(float));
////        cudaMemcpy(host_data, data, n * sizeof(float), cudaMemcpyDeviceToHost);
////
////        std::cout << "Random numbers generated on GPU:" << std::endl;
////        for (size_t i = 0; i < n; ++i) {
////            std::cout << host_data[i] << " ";
////        }
////        std::cout << std::endl;
////
////        free(host_data);
////    }
////    else {
////        std::cout << "Random numbers generated on CPU:" << std::endl;
////        for (size_t i = 0; i < n; ++i) {
////            std::cout << data[i] << " ";
////        }
////        std::cout << std::endl;
////    }
////
////    // Cleanup
////    rng.cleanup();
////
////    // Free allocated memory
////    if (use_gpu) {
////        cudaFree(data);
////    }
////    else {
////        free(data);
////    }
////
////    return 0;
////}
