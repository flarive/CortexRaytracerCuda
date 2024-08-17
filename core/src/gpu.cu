#include <iostream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <curand_kernel.h>


#include "misc/bvh_node.cuh"
#include "cameras/camera.cuh"
#include "cameras/perspective_camera.cuh"
#include "primitives/hittable_list.cuh"
#include "primitives/sphere.cuh"
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
#include "primitives/flip_normals.cuh"



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

__device__ vector3 color(const ray& r, const vector3& background, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vector3 cur_attenuation = vector3(1.0, 1.0, 1.0);
    vector3 cur_emitted = vector3(0.0, 0.0, 0.0);
    for(int i = 0; i < 100; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector3 attenuation;
            vector3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
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
        curandState local_rand_state = *rand_state;
        int i = 0;
        elist[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(vector3(0.12, 0.45, 0.15)))));
        elist[i++] = new yz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(vector3(0.65, 0.05, 0.05))));
        elist[i++] = new xz_rect(113, 443, 127, 432, 554, new diffuse_light(new solid_color_texture(vector3(1.0, 1.0, 1.0))));
        elist[i++] = new xz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73))));
        elist[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new checker_texture(
            new solid_color_texture(vector3(1, 1, 1)),
            new solid_color_texture(vector3(0, 1, 0))
        ))));
        elist[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new checker_texture(
            new solid_color_texture(vector3(1, 1, 1)),
            new solid_color_texture(vector3(0, 1, 0))
        ))));
        elist[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73)))));


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

        elist[i++] = new box(vector3(0, 0, 295), vector3(165, 330, 165), new lambertian(new solid_color_texture(vector3(0.73, 0.73, 0.73))));
        


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

        elist[i++] = new sphere(vector3(350, 50, 295), 100.0, new lambertian(*texture));


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

__global__ void render(vector3* fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = randState[pixel_index];
    vector3 col(0,0,0);
    vector3 background(0, 0, 0);

    for(int s=0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, background, world, &local_rand_state);
    }

    randState[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

void renderGPU(int nx, int ny, int ns, int tx, int ty, const char* filepath)
{
    std::cout << "Rendering " << nx << "x" << ny << " " << ns << " samples > " << filepath << std::endl;
    
    // Values
    int num_pixels = nx * ny;

    int tex_x, tex_y, tex_n;
    unsigned char *tex_data_host = stbi_load("e:\\earth_diffuse.jpg", &tex_x, &tex_y, &tex_n, 0);
    if (!tex_data_host) {
        std::cerr << "Failed to load texture." << std::endl;
        return;
    }

    unsigned char *tex_data;
    checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    image_texture**texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(image_texture*)));
    texture_init<<<1, 1>>>(tex_data, tex_x, tex_y, texture);

    // Allocating CUDA memory
    vector3* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, nx * ny * sizeof(vector3)));

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
    create_cornell_box<<<1, 1>>>(elist, eworld, cam, nx, ny, texture, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(image, nx, ny,  ns, cam, eworld, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    uint8_t* imageHost = new uint8_t[nx * ny * 3 * sizeof(uint8_t)];
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            imageHost[(ny - j - 1) * nx * 3 + i * 3] = 255.99 * image[pixel_index].r;
            imageHost[(ny - j - 1) * nx * 3 + i * 3 + 1] = 255.99 * image[pixel_index].g;
            imageHost[(ny - j - 1) * nx * 3 + i * 3 + 2] = 255.99 * image[pixel_index].b;
        }
    }
    stbi_write_png(filepath, nx, ny, 3, imageHost, nx * 3);

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(eworld));
    checkCudaErrors(cudaFree(elist));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(image));
}



void launchGPU(int nx, int ny, int ns, int tx, int ty, const char* filepath, bool quietMode)
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
    renderGPU(nx, ny, ns, tx, ty, filepath);
}