#include <curand.h>
#include <iostream>

// Structure to manage both CPU and GPU generators
struct RNG
{
    curandGenerator_t generator;  // CURAND generator
    bool use_gpu;                 // Flag to determine if we are using GPU or CPU

    // Initialize the generator based on whether we want GPU or CPU RNG
    void init(bool gpu)
    {
        use_gpu = gpu;
        if (use_gpu)
        {
            // Create GPU-based generator
            curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
            std::cout << "Initialized GPU-based random number generator." << std::endl;
        }
        else
        {
            // Create CPU-based generator
            curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
            std::cout << "Initialized CPU-based random number generator." << std::endl;
        }
    }

    // Generate random numbers on either the GPU or CPU
    void generate(float* data, size_t n)
    {
        if (use_gpu) {
            // GPU-based random number generation (on device memory)
            curandGenerateUniform(generator, data, n);
        }
        else {
            // CPU-based random number generation (on host memory)
            curandGenerateUniform(generator, data, n);
        }
    }

    // Clean up
    void cleanup()
    {
        curandDestroyGenerator(generator);
        std::cout << "Cleaned up the random number generator." << std::endl;
    }
};

//int main() {
//    // User choice for GPU or CPU random number generation
//    bool use_gpu;
//    std::cout << "Enter 1 to use GPU-based RNG, 0 for CPU-based RNG: ";
//    std::cin >> use_gpu;
//
//    size_t n = 10;  // Number of random numbers to generate
//    float* data;
//
//    RNG rng;
//    rng.init(use_gpu);
//
//    // Allocate memory based on the target (GPU or CPU)
//    if (use_gpu) {
//        // GPU allocation
//        cudaMalloc(&data, n * sizeof(float));
//    }
//    else {
//        // CPU allocation
//        data = (float*)malloc(n * sizeof(float));
//    }
//
//    // Generate random numbers
//    rng.generate(data, n);
//
//    // Copy data from GPU if needed
//    if (use_gpu) {
//        float* host_data = (float*)malloc(n * sizeof(float));
//        cudaMemcpy(host_data, data, n * sizeof(float), cudaMemcpyDeviceToHost);
//
//        std::cout << "Random numbers generated on GPU:" << std::endl;
//        for (size_t i = 0; i < n; ++i) {
//            std::cout << host_data[i] << " ";
//        }
//        std::cout << std::endl;
//
//        free(host_data);
//    }
//    else {
//        std::cout << "Random numbers generated on CPU:" << std::endl;
//        for (size_t i = 0; i < n; ++i) {
//            std::cout << data[i] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    // Cleanup
//    rng.cleanup();
//
//    // Free allocated memory
//    if (use_gpu) {
//        cudaFree(data);
//    }
//    else {
//        free(data);
//    }
//
//    return 0;
//}
