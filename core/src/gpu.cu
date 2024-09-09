#include <iostream>
//#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>

#include <curand_kernel.h>

#include "misc/vector3.cuh"
#include "misc/bvh_node.cuh"

#include "primitives/hittable_list.cuh"

#include "textures/texture.cuh"
#include "textures/solid_color_texture.cuh"
#include "textures/checker_texture.cuh"
#include "textures/image_texture.cuh"
#include "textures/bump_texture.cuh"
#include "textures/normal_texture.cuh"
#include "textures/alpha_texture.cuh"
#include "textures/emissive_texture.cuh"




#include "materials/diffuse_light.cuh"
#include "materials/diffuse_spot_light.cuh"
#include "materials/lambertian.cuh"
#include "materials/metal.cuh"
#include "materials/dielectric.cuh"
#include "materials/isotropic.cuh"



#include "primitives/aarect.cuh"
#include "primitives/box.cuh"
#include "primitives/sphere.cuh"
#include "primitives/quad.cuh"
#include "primitives/volume.cuh"
#include "primitives/torus.cuh"
#include "primitives/cylinder.cuh"
#include "primitives/cone.cuh"

#include "primitives/translate.cuh"
#include "primitives/rotate.cuh"
#include "primitives/scale.cuh"
#include "primitives/flip_normals.cuh"


#include "lights/light.cuh"
#include "lights/omni_light.cuh"
#include "lights/directional_light.cuh"
#include "lights/spot_light.cuh"

#include "cameras/camera.cuh"
#include "cameras/perspective_camera.cuh"
#include "cameras/orthographic_camera.cuh"

#include "samplers/sampler.cuh"
#include "samplers/random_sampler.cuh"

#include "utilities/bitmap_image.cuh"


#include "scenes/scene_loader.h"
#include "scenes/scene_builder.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>




// https://github.com/Belval/raytracing

bool isGpuAvailable(cudaDeviceProp& prop)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaGetDeviceProperties(&prop, deviceIndex);
        if (prop.major >= 2 && prop.minor >= 0)
        {
            printf("[INFO] Use GPU device %d %s\n", deviceIndex, prop.name);
            printf("[INFO] Number of multiprocessors on device: %d\n", prop.multiProcessorCount);
            printf("[INFO] Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
            printf("[INFO] Max grid size: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("[INFO] Max block size: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("[INFO] Max number of threads per block: %i\n", prop.maxThreadsPerBlock);

            cudaSetDevice(deviceIndex);

            return true;
        }
    }

    std::cout << "[ERROR] No Nvidia Cuda GPU device found" << std::endl;
    return false;
}


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' " << cudaGetErrorString(result) << "\n";
        // reset device before terminating
        cudaDeviceReset();
        exit(99);
    }
}

#define RND (curand_uniform(&local_rand_state))

__global__ void load_scene(hittable_list **elist, hittable_list **elights,  camera **cam, sampler **aa_sampler, int width, int height, float ratio, int spp, int sqrt_spp, image_texture** texture, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;

        //*myscene = new scene();

        *elights = new hittable_list();

        *elist = new hittable_list();

        

        (*elist)->add(new rt::flip_normals(new yz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.12, 0.45, 0.15))), "MyLeft")));
        (*elist)->add(new yz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(color(0.65, 0.05, 0.05))), "MyRight"));
        (*elist)->add(new xz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyGround"));
        (*elist)->add(new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyTop")));
        (*elist)->add(new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyBottom")));
        
        // back
        (*elist)->add(new quad(point3(0,0,555), vector3(555,0,0), vector3(0,555,0), new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyBack"));


        // box
        (*elist)->add(new rt::translate(new box(vector3(0, 0, 295), vector3(165, 330, 165), new lambertian(*texture), "MyBox"), vector3(120,0,320)));
        
        // sphere
        (*elist)->add(new sphere(vector3(350.0f, 50.0f, 295.0f), 100.0f, new lambertian(*texture), "MySphere"));

        // light
        (*elist)->add(new directional_light(point3(278, 554, 332), vector3(-305, 0, 0), vector3(0, 0, -305), 1.0f, color(10.0, 10.0, 10.0), "MyLight", true));




        // temp extract_emissive_objects
        for (int i = 0; i < (*elist)->object_count; i++)
        {
            if ((*elist)->objects[i]->getTypeID() == HittableTypeID::lightDirectionalType)
            {
                light* derived = static_cast<light*>((*elist)->objects[i]);
                if (derived)
                {
                    (*elights)->add((*elist)->objects[i]);
                }
            }
        }

        *cam = new perspective_camera();
        (*cam)->initialize(
            vector3(278, 278, -800),
            vector3(278, 278, 0),
            vector3(0, 1, 0),
            width,
            ratio,
            40.0f,
            0.0f,
            10.0f,
            0.0f,
            0.0f,
            1.0f,
            sqrt_spp);


        //*aa_sampler = new random_sampler((*cam)->get_pixel_delta_u(), (*cam)->get_pixel_delta_v(), 50);


        // calculate bounding boxes to speed up ray computing
        *elist = new hittable_list(new bvh_node((*elist)->objects, 0, (*elist)->object_count, &local_rand_state));
    }
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int maxx, int maxy, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= maxx) || (j >= maxy)) return;
    int pixel_index = j*maxx + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void texture_init(unsigned char* tex_data, int width, int height, int channels, image_texture** tex)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *tex = new image_texture(bitmap_image(tex_data, width, height, channels));
    }
}

//__global__ void render(color* fb, int width, int height, int spp, int sqrt_spp, int max_depth, hittable_list **world, hittable_list **lights, camera** cam, sampler** aa_sampler, curandState *randState)
//{
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int j = threadIdx.y + blockIdx.y * blockDim.y;
//    if ((i >= width) || (j >= height))
//        return;
//    
//        
//
//    int pixel_index = j * width + i;
//    curandState local_rand_state = randState[pixel_index];
//    color pixel_color(0, 0, 0);
//
//    for (int s_j = 0; s_j < sqrt_spp; ++s_j)
//    {
//        for (int s_i = 0; s_i < sqrt_spp; ++s_i)
//        {
//            float uniform_random = curand_uniform(&local_rand_state);
//            float gaussian_random = curand_normal(&local_rand_state);
//
//
//            // Subpixel Sampling for Anti-Aliasing
//            // Normalized Device Coordinates (NDC)
//            // By using u and v, you introduce small random perturbations within each pixel. These perturbations generate different rays that pass through slightly different positions in the scene. This increases the accuracy of the final color calculation for each pixel when averaged over multiple samples (spp or samples per pixel), producing smoother gradients and less noise.
//            float u = float(i + gaussian_random) / float(width);
//            float v = float(j + gaussian_random) / float(height);
//
//            // Stratified sampling with jittered randomness
//            //float u = (i + (s_i + curand_uniform(&local_rand_state)) / sqrt_spp) / float(width);
//            //float v = (j + (s_j + curand_uniform(&local_rand_state)) / sqrt_spp) / float(height);
//
//            
//
//
//            ray r = (*cam)->get_ray(u, v, s_i, s_j, nullptr, &local_rand_state);
//
//            // pixel color is progressively being refined
//            pixel_color += (*cam)->ray_color(r, i, j, max_depth, **world, **lights, &local_rand_state);
//        }
//    }
//
//    const color& fix = prepare_pixel_color(i, j, pixel_color, spp, true);
//
//    const interval intensity(0.000f, 0.999f);
//
//
//    randState[pixel_index] = local_rand_state;
//
//    int color_r = static_cast<int>(255.99f * intensity.clamp(fix.r()));
//    int color_g = static_cast<int>(255.99f * intensity.clamp(fix.g()));
//    int color_b = static_cast<int>(255.99f * intensity.clamp(fix.b()));
//
//
//    fb[pixel_index] = color(
//        color_r,
//        color_g,
//        color_b
//    );
//
//    printf(
//        "p %u %u %u %u %u\n",
//        i,
//        height - j - 1,
//        color_r,
//        color_g,
//        color_b
//    );
//}


__global__ void render(color* fb, int width, int height, int spp, int sqrt_spp, int max_depth, hittable_list** world, hittable_list** lights, camera** cam, sampler** aa_sampler, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;

    int pixel_index = j * width + i;
    curandState local_rand_state = randState[pixel_index];
    color pixel_color(0, 0, 0);

    float uniform_random = curand_uniform(&local_rand_state);
    //float normal_random = curand_normal(&local_rand_state);
    //int poisson_random = curand_poisson(&local_rand_state, 100.0f);

    for (int s_j = 0; s_j < sqrt_spp; ++s_j)
    {
        for (int s_i = 0; s_i < sqrt_spp; ++s_i)
        {
            // Stratified sampling within the pixel, with Sobol randomness
            float u = (i + (s_i + uniform_random) / sqrt_spp) / float(width);
            float v = (j + (s_j + uniform_random) / sqrt_spp) / float(height);

            ray r = (*cam)->get_ray(u, v, s_i, s_j, nullptr, &local_rand_state);
            pixel_color += (*cam)->ray_color(r, i, j, max_depth, max_depth, **world, **lights, &local_rand_state);
        }
    }

    const color& fix = prepare_pixel_color(i, j, pixel_color, spp, true);
    const interval intensity(0.000f, 0.999f);

    randState[pixel_index] = local_rand_state;

    int color_r = static_cast<int>(255.99f * intensity.clamp(fix.r()));
    int color_g = static_cast<int>(255.99f * intensity.clamp(fix.g()));
    int color_b = static_cast<int>(255.99f * intensity.clamp(fix.b()));

    fb[pixel_index] = color(color_r, color_g, color_b);

    printf("p %u %u %u %u %u\n", i, height - j - 1, color_r, color_g, color_b);
}


void setupCuda(const cudaDeviceProp& prop)
{
    // If you get a null pointer (either from device malloc or device new) you have run out of heap space.
    // https://forums.developer.nvidia.com/t/allocating-memory-from-device-and-cudalimitmallocheapsize/70441
    
    size_t stackSize;

    // Get the current stack size limit
    cudaError_t result1 = cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (result1 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to get stack size: " << cudaGetErrorString(result1) << std::endl;
        return;
    }

    std::cout << "[INFO] Current stack size limit: " << stackSize << " bytes" << std::endl;


    const size_t newStackSize = 4096; // Set the stack size to 1MB per thread

    cudaError_t result2 = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (result2 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set stack size: " << cudaGetErrorString(result2) << std::endl;
        return;
    }

    std::cout << "[INFO] New stack size limit: " << newStackSize << " bytes" << std::endl;



    const size_t newMallocHeapSize = size_t(1024) * size_t(1024) * size_t(1024);

    cudaError_t result3 = cudaDeviceSetLimit(cudaLimitMallocHeapSize, newMallocHeapSize);
    if (result3 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set malloc heap size: " << cudaGetErrorString(result3) << std::endl;
        return;
    }

    std::cout << "[INFO] New malloc heap limit: " << newMallocHeapSize << " bytes" << std::endl;



    const size_t newPrintfFifoSize = 10000000;

    cudaError_t result4 = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, newPrintfFifoSize);
    if (result4 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set printf fifo size: " << cudaGetErrorString(result4) << std::endl;
        return;
    }

    std::cout << "[INFO] New printf fifo size: " << newPrintfFifoSize << " bytes" << std::endl;
}

void renderGPU(const cudaDeviceProp& prop, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath)
{
    std::cout << "[INFO] Rendering " << width << "x" << height << " " << spp << " samples > " << filepath << std::endl;

    setupCuda(prop);



    float ratio = (float)height / (float)width;


    int sqrt_spp = static_cast<int>(sqrt(spp));
    
    // Values
    int num_pixels = width * height;

    int bytes_per_pixel = 3;
    int tex_x, tex_y, tex_n;
    unsigned char *tex_data_host = stbi_load("d:\\uv_mapper_no_numbers.jpg", &tex_x, &tex_y, &tex_n, bytes_per_pixel);
    if (!tex_data_host) {
        std::cerr << "[ERROR] Failed to load texture." << std::endl;
        return;
    }

    unsigned char *tex_data;
    checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));



    dim3 single_block(1, 1);
    dim3 single_thread(1, 1);



    image_texture**texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(image_texture*)));
    texture_init<<<single_block, single_thread>>>(tex_data, tex_x, tex_y, tex_n, texture);





    // Allocating CUDA memory
    color* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, width * height * sizeof(color)));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // Allocate 2nd random state to be initialized for the world creation
    rand_init<<<single_block, single_thread>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Building the world
    hittable_list **elist;
    checkCudaErrors(cudaMalloc((void**)&elist, sizeof(hittable_list*)));

    hittable_list **elights;
    checkCudaErrors(cudaMalloc((void**)&elights, sizeof(hittable_list*)));
    
    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    sampler** aa_sampler;
    checkCudaErrors(cudaMalloc((void**)&aa_sampler, sizeof(sampler*)));


    //scene** myscene;
    //checkCudaErrors(cudaMalloc((void**)&myscene, sizeof(scene*)));


    load_scene<<<single_block, single_thread>>>(elist, elights, cam, aa_sampler, width, height, ratio, spp, sqrt_spp, texture, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 render_blocks(width / tx+1, height / ty+1);
    dim3 render_threads(tx, ty);

    render_init<<<render_blocks, render_threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("[INFO] Render with %u/%u blocks of %u/%u threads\n", render_blocks.x, render_blocks.y, render_threads.x, render_threads.y);


    render<<<render_blocks, render_threads>>>(image, width, height, spp, sqrt_spp, max_depth, elist, elights, cam, aa_sampler, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // little padding to avoid remaining black zone at the end of the render preview
    for (int jj = 0; jj < 4; jj++)
    {
        for (int ii = 0; ii < width; ii++)
        {
            printf(
                "p %u %u %u %u %u\n",
                ii,
                height - jj - 1,
                0,
                0,
                0
            );
        }
    }
    


    uint8_t* imageHost = new uint8_t[width * height * 3 * sizeof(uint8_t)];
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;

            imageHost[(height - j - 1) * width * 3 + i * 3] = (size_t)image[pixel_index].r();
            imageHost[(height - j - 1) * width * 3 + i * 3 + 1] = (size_t)image[pixel_index].g();
            imageHost[(height - j - 1) * width * 3 + i * 3 + 2] = (size_t)image[pixel_index].b();
        }
    }

    stbi_write_png(filepath, width, height, 3, imageHost, width * 3);

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(elights));
    checkCudaErrors(cudaFree(elist));
    //checkCudaErrors(cudaFree(myscene));
    checkCudaErrors(cudaFree(aa_sampler));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(image));
}


void launchGPU(int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath, bool quietMode)
{
    cudaDeviceProp prop;
    
    if (!isGpuAvailable(prop))
    {
        return;
    }

    //std::cout << "Rendering222 " << nx << "x" << ny << " " << ns << " samples > " << filepath << std::endl;

    //std::cout << "[INFO] Use GPU device " << deviceIndex << " " << deviceName << std::endl;

    // https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
    // __global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
    // __device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
    // __host__ - Runs on the CPU, called from the CPU.
    // 
    // --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --std c++20 --verbose
    // --expt-relaxed-constexpr --std c++20 -Xcudafe="--diag_suppress=20012 --diag_suppress=20208" 
    //
    renderGPU(prop, width, height, spp, max_depth, tx, ty, filepath);
}


//int main(int argc, char* argv[])
//{
//    launchGPU(256, 144, 10, 2, 16, 16, "e:\\ttt2.png", true);
//}