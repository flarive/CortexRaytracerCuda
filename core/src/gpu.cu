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
#include "lights/omni_light.cuh"
#include "lights/directional_light.cuh"

#include "utilities/bitmap_image.cuh"

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
            printf("Use GPU device %d %s\n", deviceIndex, prop.name);
            printf("Number of multiprocessors on device: %d\n", prop.multiProcessorCount);
            printf("Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
            printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
            printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
            printf("Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
            printf("Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
            printf("minor-major: %d-%d\n", prop.minor, prop.major);
            printf("Warp-size: %d\n", prop.warpSize);
            printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
            printf("Concurrent computation/communication: %s\n", prop.deviceOverlap ? "yes" : "no");

            printf("Maximum size of each dimension of a grid: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("Maximum size of each dimension of a block: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("Maximum number of threads per block: %i\n", prop.maxThreadsPerBlock);

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

//__device__ color ray_color2(const ray& r, int i, int j, int depth, hittable_list& _world, hittable_list& _lights, curandState *local_rand_state)
//{
//    ray cur_ray = r;
//    color cur_attenuation = color(1.0, 1.0, 1.0);
//    color cur_emitted = color(0.0, 0.0, 0.0);
//
//    //for(int i = 0; i < 100; i++) {
//        hit_record rec;
//        if (_world.hit(cur_ray, interval(0.001f, FLT_MAX), rec, 0, local_rand_state))
//        {
//            scatter_record srec;
//            //ray scattered;
//            color attenuation;
//            color emitted = rec.mat->emitted(cur_ray, rec, rec.u, rec.v, rec.hit_point, local_rand_state);
//
//            if(rec.mat->scatter(cur_ray, _lights, rec, srec, local_rand_state))
//            {
//                cur_attenuation *= attenuation;
//                cur_emitted += emitted * cur_attenuation;
//                cur_ray = srec.skip_pdf_ray;
//            }
//            else {
//                return cur_emitted + emitted * cur_attenuation;
//            }
//        }
//        else {
//            return cur_emitted;
//        }
//    //}
//    return cur_emitted; // exceeded recursion
//}

__device__ color ray_color_iterative(const ray& r, int i, int j, int depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state)
{
    color result_color = color::black();
    ray current_ray = r;
    color current_attenuation = color(1.0f, 1.0f, 1.0f);

    for (int current_depth = depth; current_depth > 0; --current_depth)
    {
        hit_record rec;
        if (!_world.hit(current_ray, interval(SHADOW_ACNE_FIX, FLT_MAX), rec, current_depth, local_rand_state))
        {
            break;
        }

        scatter_record srec;
        color emitted = rec.mat->emitted(current_ray, rec, rec.u, rec.v, rec.hit_point, local_rand_state);

        if (!rec.mat->scatter(current_ray, _lights, rec, srec, local_rand_state))
        {
            result_color += current_attenuation * emitted;
            break;
        }

        hittable_pdf hpdf(_lights, rec.hit_point);
        mixture_pdf mpdf(&hpdf, srec.pdf_ptr);

        ray scattered = ray(rec.hit_point, mpdf.generate(srec, local_rand_state), current_ray.time());
        float pdf_val = mpdf.value(scattered.direction(), local_rand_state);
        float scattering_pdf = rec.mat->scattering_pdf(current_ray, rec, scattered);

        result_color += current_attenuation * emitted;
        current_attenuation *= srec.attenuation * scattering_pdf / pdf_val;

        current_ray = scattered;
    }

    return result_color;
}

__device__ color ray_color(const ray& r, int i, int j, int depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state)
{
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
    {
        return color::black();// background_color;
    }

    hit_record rec;

    vector3 unit_dir = unit_vector(r.direction());

    // If the ray hits nothing, return the background color.
    // 0.001 is to fix shadow acne interval
    if (!_world.hit(r, interval(SHADOW_ACNE_FIX, FLT_MAX), rec, depth, local_rand_state))
    {
        //if (background_texture)
        //{
        //    return get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
        //}
        //else
        //{
        //    return background_color;
        //}

        return color::black();
    }

    // ray hit a world object
    scatter_record srec;
    color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.hit_point, local_rand_state);

    // hack for invisible primitives (such as lights)
    if (color_from_emission.a() == 0.0f)
    {
        // rethrow a new ray
        _world.hit(r, interval(rec.t + 0.001f, FLT_MAX), rec, depth, local_rand_state);
    }

    if (!rec.mat->scatter(r, _lights, rec, srec, local_rand_state))
    {
        return color_from_emission;
    }

    

    if (_lights.object_count == 0)
    {
        // no lights
        // no importance sampling
        return srec.attenuation * ray_color(srec.skip_pdf_ray, i, j, depth - 1, _world, _lights, local_rand_state);
    }

    // no importance sampling
    if (srec.skip_pdf)
    {
        return srec.attenuation * ray_color(srec.skip_pdf_ray, i, j, depth - 1, _world, _lights, local_rand_state);
    }

    //hittable_pdf* hpdf = new hittable_pdf(_lights, rec.hit_point);


    //mixture_pdf* mpdf;

    //if (background_texture && background_iskybox)
    //{
    //    mixture_pdf p_objs(light_ptr, srec.pdf_ptr, 0.5f);
    //    p = mixture_pdf(new mixture_pdf(p_objs), background_pdf, 0.8f);
    //}
    //else
    //{
    //mpdf = new mixture_pdf(hpdf, srec.pdf_ptr);
    //}

    hittable_pdf hpdf(_lights, rec.hit_point);
    mixture_pdf mpdf(&hpdf, srec.pdf_ptr);


    ray scattered = ray(rec.hit_point, mpdf.generate(srec, local_rand_state), r.time());
    float pdf_val = mpdf.value(scattered.direction(), local_rand_state);
    float scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

    color final_color(0,0,0);

    //if (background_texture)
    //{
        // with background image
        //bool double_sided = false;
        //if (rec.mat->has_alpha_texture(double_sided))
        //{
        //    // render transparent object (having an alpha texture)
        //    color background_behind = rec.mat->get_diffuse_pixel_color(rec);

        //    ray ray_behind(rec.hit_point, r.direction(), r.x, r.y, r.time());
        //    color background_infrontof = ray_color(ray_behind, depth - 1, _scene, local_rand_state);

        //    hit_record rec_behind;
        //    if (_scene.get_world().hit(ray_behind, interval(0.001f, INFINITY), rec_behind, depth, local_rand_state))
        //    {
        //        // another object is behind the alpha textured object, display it behind
        //        scatter_record srec_behind;

        //        if (double_sided)
        //        {
        //            if (rec_behind.mat->scatter(ray_behind, _scene.get_emissive_objects(), rec_behind, srec_behind, local_rand_state))
        //            {
        //                final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
        //            }
        //        }
        //        else
        //        {
        //            if (rec_behind.mat->scatter(ray_behind, _scene.get_emissive_objects(), rec_behind, srec_behind, local_rand_state) && rec.front_face)
        //            {
        //                final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
        //            }
        //            else
        //            {
        //                final_color = background_infrontof;
        //            }
        //        }
        //    }
        //    else
        //    {
        //        // no other object behind the alpha textured object, just display background image
        //        if (double_sided)
        //        {
        //            final_color = color::blend_colors(color_from_emission + background_behind, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), depth - 1, _scene, local_rand_state), srec.alpha_value);
        //        }
        //        else
        //        {
        //            final_color = get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
        //        }
        //    }
        //}
        //else
        //{
        //    // render opaque object
        //    color color_from_scatter = ray_color(scattered, depth - 1, _scene, local_rand_state) / pdf_val;
        //    final_color = color_from_emission + srec.attenuation * scattering_pdf * color_from_scatter;
        //}
    //}
    //else
    //{
        // with background color
        color sample_color = ray_color(scattered, i, j, depth - 1, _world, _lights, local_rand_state);
        color color_from_scatter = (srec.attenuation * scattering_pdf * sample_color) / pdf_val;

        //bool double_sided = false;
        //if (rec.mat->has_alpha_texture(double_sided))
        //{
        //    // render transparent object (having an alpha texture)
        //    final_color = color::blend_colors(color_from_emission + color_from_scatter, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), depth - 1, _world, _lights, local_rand_state), srec.alpha_value);
        //}
        //else
        //{
            // render opaque object
            final_color = color_from_emission + color_from_scatter;
        //}
    //}

    //delete hpdf;
    //delete mpdf;
     
    return final_color;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_cornell_box(hittable_list **elist, hittable_list **elights,  camera **cam, int width, int height, float ratio, int spp, int sqrt_spp, image_texture** texture, curandState *rand_state)
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
        (*elist)->add(new rt::translate(new box(vector3(0, 0, 295), vector3(165, 330, 165), new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyBox"), vector3(120,0,320)));
        

        // sphere
        (*elist)->add(new sphere(vector3(350.0f, 50.0f, 295.0f), 100.0f, new lambertian(*texture), "MySphere"));

        // light
        (*elist)->add(new directional_light(point3(278, 554, 332), vector3(-305, 0, 0), vector3(0, 0, -305), 1.0f, color(10.0, 10.0, 10.0), "MyLight", false));




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
            1.0f,
            sqrt_spp);

        //printf("test %i/%i\n", (*elist)->object_count, (*elist)->object_capacity);

        //for (int i = 0; i < (*elist)->object_count; i++)
        //    printf("test obj %i %s\n", (*elist)->objects[i]->getTypeID(), (*elist)->objects[i]->getName());

        // calculate bounding boxes to speed up ray computing
        *elist = new hittable_list(new bvh_node((*elist)->objects, 0, (*elist)->object_count, &local_rand_state));


        //(*myscene)->set(*elist);
        //(*myscene)->set_camera(*cam);
        //(*myscene)->extract_emissive_objects();
        //(*myscene)->build_optimized_world(local_rand_state);
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

__global__ void render(color* fb, int width, int height, int spp, int sqrt_spp, int max_depth, hittable_list **world, hittable_list **lights, camera** cam, curandState *randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;

    int pixel_index = j* width + i;
    curandState local_rand_state = randState[pixel_index];
    color pixel_color(0, 0, 0);

    //if (threadIdx.x == 0 && blockIdx.x == 0) {
    //    printf("Scanlines remaining: %i\n", (height - j));
    //}

    // new
    for (int s_j = 0; s_j < sqrt_spp; ++s_j)
    {
        for (int s_i = 0; s_i < sqrt_spp; ++s_i)
        {
            float u = float(i + curand_uniform(&local_rand_state)) / float(width);
            float v = float(j + curand_uniform(&local_rand_state)) / float(height);

            ray r = (*cam)->get_ray(u, v, s_i, s_j, nullptr, &local_rand_state);

            // pixel color is progressively being refined
            pixel_color += ray_color_iterative(r, i, j, max_depth, **world, **lights, &local_rand_state);
        }
    }


    //for (int k = 0; k < 2; k++)
    //{
    //    ray r = (*cam)->get_ray(i, j, k, k, nullptr, &local_rand_state);

    //    // pixel color is progressively being refined
    //    pixel_color += ray_color(r, i, j, max_depth, **world, **lights, &local_rand_state);
    //}

    // old
    //for(int s=0; s < spp; s++)
    //{
    //    float u = float(i + curand_uniform(&local_rand_state)) / float(width);
    //    float v = float(j + curand_uniform(&local_rand_state)) / float(height);
    //    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    //    pixel_color += get_color(r, background, world, lights, &local_rand_state);
    //}

    randState[pixel_index] = local_rand_state;
    //pixel_color /= float(spp);
    //pixel_color[0] = sqrt(pixel_color[0]);
    //pixel_color[1] = sqrt(pixel_color[1]);
    //pixel_color[2] = sqrt(pixel_color[2]);
    fb[pixel_index] = pixel_color;
}

void setupCuda(const cudaDeviceProp& prop)
{
    
    // If you get a null pointer (either from device malloc or device new) you have run out of heap space.
    // https://forums.developer.nvidia.com/t/allocating-memory-from-device-and-cudalimitmallocheapsize/70441
    
    size_t stackSize;

    // Get the current stack size limit
    cudaError_t result1 = cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (result1 != cudaSuccess) {
        std::cerr << "Failed to get stack size: " << cudaGetErrorString(result1) << std::endl;
        return;
    }

    std::cout << "Current stack size limit: " << stackSize << " bytes" << std::endl;


    const size_t newStackSize = 4096; // Set the stack size to 1MB per thread

    cudaError_t result2 = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (result2 != cudaSuccess) {
        std::cerr << "Failed to set stack size: " << cudaGetErrorString(result2) << std::endl;
        return;
    }

    std::cout << "New stack size limit: " << newStackSize << " bytes" << std::endl;



    const size_t newMallocHeapSize = size_t(1024) * size_t(1024) * size_t(1024);

    cudaError_t result3 = cudaDeviceSetLimit(cudaLimitMallocHeapSize, newMallocHeapSize);
    if (result3 != cudaSuccess) {
        std::cerr << "Failed to set malloc heap size: " << cudaGetErrorString(result3) << std::endl;
        return;
    }

    std::cout << "New malloc heap limit: " << newMallocHeapSize << " bytes" << std::endl;


    // cuda initialization via cudaMalloc
    //size_t limit = 0;

    //cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    //printf("cudaLimitStackSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    //printf("cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    //printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

    //std::cout << "default settings of cuda context" << std::endl;
    //
    //limit = 10;

    //cudaDeviceSetLimit(cudaLimitStackSize, limit);
    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, limit);
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);

    //std::cout << "set limit to 10 for all settings" << std::endl;
    //

    //limit = 0;

    //cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    //printf("New cudaLimitStackSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
    //printf("New cudaLimitPrintfFifoSize: %u\n", (unsigned)limit);
    //cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    //printf("New cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
}

void renderGPU(const cudaDeviceProp& prop, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath)
{
    std::cout << "Rendering " << width << "x" << height << " " << spp << " samples > " << filepath << std::endl;

    setupCuda(prop);



    float ratio = (float)height / (float)width;


    int sqrt_spp = static_cast<int>(sqrt(spp));
    
    // Values
    int num_pixels = width * height;

    int tex_x, tex_y, tex_n;
    unsigned char *tex_data_host = stbi_load("E:\\earth_diffuse.jpg", &tex_x, &tex_y, &tex_n, 0);
    if (!tex_data_host) {
        std::cerr << "Failed to load texture." << std::endl;
        return;
    }

    unsigned char *tex_data;
    checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));



    //dim3 init_blocks(1, 1);
    //dim3 init_threads(1, 1);



    image_texture**texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(image_texture*)));
    texture_init<<<1, 1 >>>(tex_data, tex_x, tex_y, tex_n, texture);





    // Allocating CUDA memory
    color* image;
    checkCudaErrors(cudaMallocManaged((void**)&image, width * height * sizeof(color)));

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
    hittable_list **elist;
    //int num_entity = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&elist, sizeof(hittable_list*)));

    //hittable_list **eworld;
    //checkCudaErrors(cudaMalloc((void**)&eworld, sizeof(hittable_list*)));

    hittable_list **elights;
    checkCudaErrors(cudaMalloc((void**)&elights, sizeof(hittable_list*)));
    
    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    //scene** myscene;
    //checkCudaErrors(cudaMalloc((void**)&myscene, sizeof(scene*)));


    create_cornell_box<<<1, 1>>>(elist, elights, cam, width, height, ratio, spp, sqrt_spp, texture, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 render_blocks(width / tx+1, height / ty+1);
    dim3 render_threads(tx, ty);

    render_init<<<render_blocks, render_threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Render with %u/%u blocks of %u/%u threads", render_blocks.x, render_blocks.y, render_threads.x, render_threads.y);


    render<<<render_blocks, render_threads>>>(image, width, height, spp, sqrt_spp, max_depth, elist, elights, cam, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    static const interval intensity(0.000f, 0.999f);

    uint8_t* imageHost = new uint8_t[width * height * 3 * sizeof(uint8_t)];
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;

            color fix = color::prepare_pixel_color(i, j, image[pixel_index], spp, false);

            //imageHost[j * width * 3 + i * 3] = (size_t)(256.0f * intensity.clamp(fix.r()));
            //imageHost[j * width * 3 + i * 3 + 1] = (size_t)(256.0f * intensity.clamp(fix.g()));
            //imageHost[j * width * 3 + i * 3 + 2] = (size_t)(256.0f * intensity.clamp(fix.b()));

            imageHost[(height - j - 1) * width * 3 + i * 3] = (size_t)(255.99f * intensity.clamp(fix.r()));
            imageHost[(height - j - 1) * width * 3 + i * 3 + 1] = (size_t)(255.99f * intensity.clamp(fix.g()));
            imageHost[(height - j - 1) * width * 3 + i * 3 + 2] = (size_t)(255.99f * intensity.clamp(fix.b()));
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