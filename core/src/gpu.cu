#include <iostream>
//#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>

#include <curand_kernel.h>

#include "misc/vector3.cuh"
#include "misc/bvh_node.cuh"
#include "cameras/camera.cuh"
#include "cameras/perspective_camera.cuh"
#include "primitives/hittable_list.cuh"
#include "primitives/sphere.cuh"
#include "primitives/quad.cuh"
#include "primitives/aarect.cuh"
#include "materials/diffuse_light.cuh"
#include "primitives/moving_sphere.cuh"
#include "materials/lambertian.cuh"
#include "materials/metal.cuh"
#include "materials/dielectric.cuh"
#include "textures/texture.cuh"
#include "textures/solid_color_texture.cuh"
#include "textures/checker_texture.cuh"
#include "textures/image_texture.cuh"
#include "primitives/box.cuh"

#include "materials/isotropic.cuh"
#include "primitives/volume.cuh"

#include "primitives/translate.cuh"
#include "primitives/rotate.cuh"
#include "primitives/scale.cuh"
#include "primitives/flip_normals.cuh"

#include "lights/light.cuh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>




// https://github.com/Belval/raytracing

bool isGpuAvailable()
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceProperties.major >= 2 && deviceProperties.minor >= 0)
        {
            std::cout << "Use GPU device " << deviceIndex << " " << deviceProperties.name << std::endl;
            
            cudaSetDevice(deviceIndex);

            return true;
        }
    }

    std::cout << "Use Nvidia Cuda GPU device found" << std::endl;
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

__device__ vector3 get_color(const ray& r, const vector3& background, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vector3 cur_attenuation = vector3(1.0, 1.0, 1.0);
    vector3 cur_emitted = vector3(0.0, 0.0, 0.0);
    for(int i = 0; i < 100; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec, 0, local_rand_state)) {
            ray scattered;
            vector3 attenuation;
            vector3 emitted = rec.mat->emitted(rec.u, rec.v, rec.hit_point);
            if(rec.mat->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_emitted += emitted * cur_attenuation;
                cur_ray = scattered;
            }
            else {
                return cur_emitted + emitted * cur_attenuation;
            }
        }
        else {
            return cur_emitted;
        }
    }
    return cur_emitted; // exceeded recursion
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_cornell_box(hittable **elist, hittable **eworld, camera **cam, int nx, int ny, image_texture** texture, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //curandState local_rand_state = *rand_state;
        int i = 0;
        elist[i++] = new rt::flip_normals(new yz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(vector3(0.12, 0.45, 0.15)))));
        elist[i++] = new yz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(vector3(0.65, 0.05, 0.05))));
        elist[i++] = new xz_rect(113, 443, 127, 432, 554, new diffuse_light(new solid_color_texture(vector3(1.0, 1.0, 1.0))));
        elist[i++] = new xz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73))));
        elist[i++] = new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new checker_texture(
            new solid_color_texture(vector3(1, 1, 1)),
            new solid_color_texture(vector3(0, 1, 0))
        ))));
        elist[i++] = new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new checker_texture(
            new solid_color_texture(vector3(1, 1, 1)),
            new solid_color_texture(vector3(0, 1, 0))
        ))));
        
        // back
        //elist[i++] = new xy_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73))));
        elist[i++] = new quad(point3(0,0,555), vector3(555,0,0), vector3(0,555,0), new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73))));


        /*elist[i++] = new ConstantMedium(
            new Translate(
                new RotateY(
                    new Box(vector3(0, 0, 0), vector3(165, 330, 165), new Lambertian(new ConstantTexture(vector3(0.73, 0.73, 0.73)))),
                    15
                ),
                vector3(265, 0, 295)
            ),
            0.05,
            new ConstantTexture(vector3(0, 0, 0)),
            &local_rand_state
        );*/

        elist[i++] = new rt::translate(new box(vector3(0, 0, 295), vector3(165, 330, 165), new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73)))), vector3(120,0,320));
        


        //elist[i++] = new ConstantMedium(
        //    new Translate(
        //        new RotateY(
        //            new Box(vector3(0, 0, 0), vector3(165, 165, 165), new Lambertian(new ConstantTexture(vector3(0.73, 0.73, 0.73)))),
        //            -18
        //        ),
        //        vector3(130, 0, 65)
        //    ),
        //    0.01,
        //    new ConstantTexture(vector3(0.8, 0.8, 0.8)),
        //    &local_rand_state
        //);

        elist[i++] = new sphere(vector3(350.0f, 50.0f, 295.0f), 100.0f, new lambertian(*texture), "Sphere1");


        *eworld = new hittable_list(elist, i);

        vector3 lookfrom(278, 278, -800);
        vector3 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;

        *cam = new perspective_camera();
        (*cam)->initialize(lookfrom,
            lookat,
            vector3(0, 1, 0),
            40.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0,
            1.0);
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

__global__ void texture_init(unsigned char* tex_data, int nx, int ny, image_texture** tex)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *tex = new image_texture(tex_data, nx, ny);
    }
}

__global__ void render(vector3* fb, int width, int height, int spp, int sqrt_spp, int max_depth, camera **cam, hittable **world, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;
    int pixel_index = j* width + i;
    curandState local_rand_state = randState[pixel_index];
    vector3 pixel_color(0.0f, 0.0f, 0.0f);
    vector3 background(0, 0, 0);

    // new
    //for (int s_j = 0; s_j < sqrt_spp; ++s_j)
    //{
    //    for (int s_i = 0; s_i < sqrt_spp; ++s_i)
    //    {
    //        float u = float(i + curand_uniform(&local_rand_state)) / float(width);
    //        float v = float(j + curand_uniform(&local_rand_state)) / float(height);

    //        // new
    //        ray r = (*cam)->get_ray(u, v, s_i, s_j, nullptr, &local_rand_state);
    //        // pixel color is progressively being refined
    //        pixel_color += (*cam)->ray_color(r, max_depth, world);
    //    }
    //}

    // old
    for(int s=0; s < spp; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(width);
        float v = float(j + curand_uniform(&local_rand_state)) / float(height);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        pixel_color += get_color(r, background, world, &local_rand_state);
    }

    randState[pixel_index] = local_rand_state;
    pixel_color /= float(spp);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);
    fb[pixel_index] = pixel_color;
}

void renderGPU(int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath)
{
    std::cout << "Rendering " << width << "x" << height << " " << spp << " samples > " << filepath << std::endl;

    int sqrt_spp = static_cast<int>(sqrt(spp));
    
    // Values
    int num_pixels = width * height;

    int tex_x, tex_y, tex_n;
    unsigned char *tex_data_host = stbi_load("e:\\earth_diffuse.jpg", &tex_x, &tex_y, &tex_n, 0);
    if (!tex_data_host) {
        std::cerr << "Failed to load texture." << std::endl;
        return;
    }

    size_t stackSize;

    // Get the current stack size limit
    cudaError_t result1 = cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (result1 != cudaSuccess) {
        std::cerr << "Failed to get stack size: " << cudaGetErrorString(result1) << std::endl;
        return;
    }

    std::cout << "Current stack size limit: " << stackSize << " bytes" << std::endl;


    size_t newStackSize = 2048; // Set the stack size to 1MB per thread

    cudaError_t result2 = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (result2 != cudaSuccess) {
        std::cerr << "Failed to set stack size: " << cudaGetErrorString(result2) << std::endl;
        return;
    }

    std::cout << "New stack size limit: " << newStackSize << " bytes" << std::endl;


    unsigned char *tex_data;
    checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    image_texture**texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(image_texture*)));
    texture_init<<<1, 1>>>(tex_data, tex_x, tex_y, texture);

    // Allocating CUDA memory
    vector3* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, width * height * sizeof(vector3)));

    // Allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // Allocate 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Building the world
    hittable **elist;
    int num_entity = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&elist, num_entity * sizeof(hittable*)));
    hittable **eworld;
    checkCudaErrors(cudaMalloc((void **)&eworld, sizeof(hittable*)));
    camera** cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera*)));
    create_cornell_box<<<1, 1>>>(elist, eworld, cam, width, height, texture, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(width / tx+1, height / ty+1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(image, width, height, spp, sqrt_spp, max_depth, cam, eworld, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    uint8_t* imageHost = new uint8_t[width * height * 3 * sizeof(uint8_t)];
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            imageHost[(height - j - 1) * width * 3 + i * 3] = 255.99f * image[pixel_index].r;
            imageHost[(height - j - 1) * width * 3 + i * 3 + 1] = 255.99f * image[pixel_index].g;
            imageHost[(height - j - 1) * width * 3 + i * 3 + 2] = 255.99f * image[pixel_index].b;
        }
    }
    stbi_write_png(filepath, width, height, 3, imageHost, width * 3);

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(eworld));
    checkCudaErrors(cudaFree(elist));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(image));
}



void launchGPU(int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath, bool quietMode)
{
    if (!isGpuAvailable())
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
    renderGPU(width, height, spp, max_depth, tx, ty, filepath);
}