#pragma once

#define TINYOBJLOADER_IMPLEMENTATION
#include "obj/tinyobjloader.hpp"

#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"
#include "../primitives/triangle.cuh"
#include "../materials/material.cuh"
#include "../materials/phong.cuh"
#include "../misc/vector3.cuh"
#include "../textures/solid_color_texture.cuh"
#include "../textures/bump_texture.cuh"
#include "../textures/normal_texture.cuh"
#include "../textures/displacement_texture.cuh"
#include "../misc/bvh_node.cuh"

// Removed std::array include
// #include <array>



// Structure to hold the mesh data (vertices, normals, indices)
// Structure to hold mesh data for a single OBJ file
struct obj_mesh_data {
    const char* name;
    
    float* vertices;
    float* normals;
    int* indices;
    float* texcoords;
    int* num_face_vertices;

    unsigned int num_vertices;
    unsigned int num_normals;
    unsigned int num_indices;
    unsigned int num_texcoords;

    unsigned int num_face_vertices_size;
    unsigned int count_total_vertices;
};


//struct obj_meshes_data {
//    obj_mesh_data** shapes;
//
//    int num_shapes;
//    int num_materials;
//};

// Structure to hold data for multiple meshes (multiple OBJ files)
struct obj_scene_data {
    obj_mesh_data** meshes;
    unsigned int num_meshes;
};




class mesh_loader
{
public:

    mesh_loader()
    {
    }

    typedef struct
    {
        // Replace std::vector with raw pointers and counts
        tinyobj::shape_t* shapes;
        size_t num_shapes;
        tinyobj::material_t* materials;
        size_t num_materials;
        tinyobj::attrib_t attributes;
    } mesh_data;


    __host__ __device__  static bool load_model_from_file(const char* filepath, mesh_data& mesh);

    //__host__ __device__  static hittable* convert_model_from_file(mesh_data& data, material* model_material, bool use_mtl, bool shade_smooth, const char* name = "");

    __device__ static hittable* convert_model_from_file(obj_mesh_data* data, material* model_material, bool use_mtl, bool shade_smooth, const char* name);

    __host__ __device__  static color get_color(tinyobj::real_t* raws);

    __host__ __device__  static material* get_mtl_mat(const tinyobj::material_t& reader_mat);

    __host__ __device__  static void computeTangentBasis(vector3 vertices[3], vector2 uvs[3], vector3 normals[3], vector3 tangents[3], vector3 bitangents[3]);

    __host__ __device__  static void applyDisplacement(mesh_data& data, displacement_texture* tex);
};

__host__ __device__ bool mesh_loader::load_model_from_file(const char* filepath, mesh_data& mesh)
{
    printf("[INFO] Loading obj file %s\n", filepath);

    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath, reader_config))
    {
        if (!reader.Error().empty())
        {
            printf("[ERROR] Loading obj file error: %s\n", reader.Error().c_str());
        }
        exit(1);
    }

    if (!reader.Warning().empty())
    {
        printf("[WARN] Loading obj file warning: %s\n", reader.Warning().c_str());
    }

    //try
    //{
        // Copy shapes and materials into raw arrays
        std::vector<tinyobj::shape_t> shape_vector = reader.GetShapes();
        std::vector<tinyobj::material_t> material_vector = reader.GetMaterials();

        mesh.num_shapes = shape_vector.size();
        mesh.shapes = new tinyobj::shape_t[mesh.num_shapes];
        for (size_t i = 0; i < mesh.num_shapes; ++i)
        {
            mesh.shapes[i] = shape_vector[i];
        }

        mesh.num_materials = material_vector.size();
        mesh.materials = new tinyobj::material_t[mesh.num_materials];
        for (size_t i = 0; i < mesh.num_materials; ++i)
        {
            mesh.materials[i] = material_vector[i];
        }

        mesh.attributes = reader.GetAttrib();

        return true;
    //}
    //catch (const std::exception&)
    //{
    //    printf("[ERROR] Loading obj file failed: %s\n", reader.Error().c_str());
    //    exit(1);
    //}

    //return false;
}

//__host__ __device__ hittable* mesh_loader::convert_model_from_file(mesh_data& data, material* model_material, bool use_mtl, bool shade_smooth, const char* name)
//{
//    hittable_list model_output;
//
//    int seed = 789999;
//    thrust::minstd_rand rng(seed);
//
//    printf("[INFO] Start building obj file (%zu objects found)\n", data.num_shapes);
//
//    // Replace std::vector with raw array
//    material** converted_mats = nullptr;
//    if (data.num_materials > 0)
//    {
//        converted_mats = new material * [data.num_materials];
//        for (size_t i = 0; i < data.num_materials; ++i)
//        {
//            converted_mats[i] = get_mtl_mat(data.materials[i]);
//        }
//    }
//
//    const bool use_mtl_file = use_mtl && (data.num_materials != 0);
//
//    // Loop over shapes
//    for (size_t s = 0; s < data.num_shapes; s++)
//    {
//        if (use_mtl_file)
//        {
//            // Handle displacement textures if necessary
//        }
//        else if (model_material && model_material->has_displace_texture())
//        {
//            // Handle displacement textures if necessary
//        }
//
//        hittable_list shape_triangles;
//
//        size_t index_offset = 0;
//
//        tinyobj::shape_t& shape = data.shapes[s];
//        size_t num_faces = shape.mesh.num_face_vertices.size();
//
//        // Loop over faces (polygons)
//        for (size_t f = 0; f < num_faces; f++)
//        {
//            const int fv = 3; // Assuming triangulated faces
//
//            // Ensure face has 3 vertices
//            assert(shape.mesh.num_face_vertices[f] == fv);
//
//            // Replace std::array with raw arrays
//            vector3 tri_v[3];
//            vector3 tri_vn[3];
//            vector2 tri_uv[3];
//
//            // Loop over vertices in the face
//            for (size_t v = 0; v < 3; v++)
//            {
//                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
//                tinyobj::real_t vx = data.attributes.vertices[3 * idx.vertex_index + 0];
//                tinyobj::real_t vy = data.attributes.vertices[3 * idx.vertex_index + 1];
//                tinyobj::real_t vz = data.attributes.vertices[3 * idx.vertex_index + 2];
//
//                tri_v[v] = vector3(vx, vy, vz);
//
//                if (idx.normal_index >= 0)
//                {
//                    tinyobj::real_t nx = data.attributes.normals[3 * idx.normal_index + 0];
//                    tinyobj::real_t ny = data.attributes.normals[3 * idx.normal_index + 1];
//                    tinyobj::real_t nz = data.attributes.normals[3 * idx.normal_index + 2];
//
//                    tri_vn[v] = vector3(nx, ny, nz);
//                }
//
//                if (idx.texcoord_index >= 0)
//                {
//                    tinyobj::real_t tu = data.attributes.texcoords[2 * idx.texcoord_index + 0];
//                    tinyobj::real_t tv = data.attributes.texcoords[2 * idx.texcoord_index + 1];
//
//                    tri_uv[v] = vector2(tu, tv);
//                }
//            }
//
//            // Replace std::array with raw arrays
//            vector3 tri_tan[3];
//            vector3 tri_bitan[3];
//            computeTangentBasis(tri_v, tri_uv, tri_vn, tri_tan, tri_bitan);
//
//            material* tri_mat;
//            if (use_mtl_file)
//            {
//                tri_mat = converted_mats[shape.mesh.material_ids[f]];
//            }
//            else
//            {
//                tri_mat = model_material;
//            }
//
//            shape_triangles.add(new triangle(
//                tri_v[0], tri_v[1], tri_v[2],
//                tri_vn[0], tri_vn[1], tri_vn[2],
//                tri_uv[0], tri_uv[1], tri_uv[2],
//                tri_tan[0], tri_tan[1], tri_tan[2],
//                tri_bitan[0], tri_bitan[1], tri_bitan[2],
//                shade_smooth, tri_mat));
//
//            index_offset += fv;
//        }
//
//        printf("[INFO] Parsing obj file (object name %s / %d vertex / %zu faces)\n",
//            shape.name.c_str(), (int)(data.attributes.vertices.size() / 3), num_faces);
//
//        model_output.add(new bvh_node(shape_triangles.objects, 0, shape_triangles.object_count, rng, name));
//    }
//
//    printf("[INFO] End building obj file\n");
//
//    if (converted_mats)
//    {
//        delete[] converted_mats;
//    }
//
//    return new bvh_node(model_output.objects, 0, model_output.object_count, rng, name);
//}



__device__ hittable* mesh_loader::convert_model_from_file(obj_mesh_data* data, material* model_material, bool use_mtl, bool shade_smooth, const char* name)
{
    hittable_list model_output;


    if (data == nullptr)
    {
        printf("invalid data !!!\n");
    }

    int num_shapes = 1;

    int seed = 789999;
    thrust::minstd_rand rng(seed);

    printf("[INFO] Start building obj file (%d objects found)\n", num_shapes);

    // Replace std::vector with raw array
    //material** converted_mats = nullptr;
    //if (data.num_materials > 0)
    //{
    //    converted_mats = new material * [data.num_materials];
    //    for (size_t i = 0; i < data.num_materials; ++i)
    //    {
    //        converted_mats[i] = get_mtl_mat(data.materials[i]);
    //    }
    //}

    const bool use_mtl_file = false;// use_mtl && (data.num_materials != 0);

    // Loop over shapes
    for (size_t s = 0; s < num_shapes; s++)
    {
        if (use_mtl_file)
        {
            // Handle displacement textures if necessary
        }
        else if (model_material && model_material->has_displace_texture())
        {
            // Handle displacement textures if necessary
        }

        hittable_list* shape_triangles = new hittable_list();

        size_t index_offset = 0;

        if (data != nullptr)
        {
            //tinyobj::shape_t& shape = data.shapes[s];
            size_t num_faces = data->num_face_vertices_size;

            // Loop over faces (polygons)
            for (size_t f = 0; f < num_faces; f++)
            {
                const int fv = 3; // Assuming triangulated faces

                // Ensure face has 3 vertices
                assert(data->num_face_vertices[f] == fv);

                // Replace std::array with raw arrays
                vector3 tri_v[3];
                vector3 tri_vn[3];
                vector2 tri_uv[3];

                // Loop over vertices in the face
                for (size_t v = 0; v < 3; v++)
                {
                    //tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    int idx = data->indices[index_offset + v];


                    //tinyobj::real_t vx = data.attributes.vertices[3 * idx.vertex_index + 0];
                    //tinyobj::real_t vy = data.attributes.vertices[3 * idx.vertex_index + 1];
                    //tinyobj::real_t vz = data.attributes.vertices[3 * idx.vertex_index + 2];
                    auto vx = data->vertices[3 * idx + 0];
                    auto vy = data->vertices[3 * idx + 1];
                    auto vz = data->vertices[3 * idx + 2];

                    tri_v[v] = vector3(vx, vy, vz);

                    /*if (idx.normal_index >= 0)
                    {*/
                    //tinyobj::real_t nx = data.attributes.normals[3 * idx.normal_index + 0];
                    //tinyobj::real_t ny = data.attributes.normals[3 * idx.normal_index + 1];
                    //tinyobj::real_t nz = data.attributes.normals[3 * idx.normal_index + 2];
                    auto nx = data->normals[3 * idx + 0];
                    auto ny = data->normals[3 * idx + 1];
                    auto nz = data->normals[3 * idx + 2];

                    tri_vn[v] = vector3(nx, ny, nz);
                    //}

                    //if (idx.texcoord_index >= 0)
                    //{
                        //tinyobj::real_t tu = data.attributes.texcoords[2 * idx.texcoord_index + 0];
                        //tinyobj::real_t tv = data.attributes.texcoords[2 * idx.texcoord_index + 1];
                    auto tu = data->texcoords[3 * idx + 0];
                    auto tv = data->texcoords[3 * idx + 1];

                    tri_uv[v] = vector2(tu, tv);
                    //}
                }

                // Replace std::array with raw arrays
                vector3 tri_tan[3];
                vector3 tri_bitan[3];
                computeTangentBasis(tri_v, tri_uv, tri_vn, tri_tan, tri_bitan);

                material* tri_mat;
                if (use_mtl_file)
                {
                    //tri_mat = converted_mats[shape.mesh.material_ids[f]];
                }
                else
                {
                    tri_mat = model_material;
                }

                //printf("shape_triangles.add(new triangle count = %u\n", shape_triangles->object_count);


                shape_triangles->add(new triangle(
                    tri_v[0], tri_v[1], tri_v[2],
                    tri_vn[0], tri_vn[1], tri_vn[2],
                    tri_uv[0], tri_uv[1], tri_uv[2],
                    tri_tan[0], tri_tan[1], tri_tan[2],
                    tri_bitan[0], tri_bitan[1], tri_bitan[2],
                    shade_smooth, tri_mat));

                index_offset += fv;
            }
        }

        //printf("[INFO] Parsing obj file (object name %s / %d vertex / %d faces)\n", data->name, data->count_total_vertices, data->num_face_vertices_size);


        //printf("shape_triangles.object_count = %u\n", shape_triangles->object_count);

        auto n = new bvh_node(shape_triangles->objects, 0, shape_triangles->object_count, rng, name);

        model_output.add(n);
    }

    printf("[INFO] End building obj file\n");

    //if (converted_mats)
    //{
    //    delete[] converted_mats;
    //}

    return new bvh_node(model_output.objects, 0, model_output.object_count, rng, name);
}

__host__ __device__ void mesh_loader::computeTangentBasis(vector3 vertices[3], vector2 uvs[3], vector3 normals[3], vector3 tangents[3], vector3 bitangents[3])
{
    // Shortcuts for vertices
    vector3& v0 = vertices[0];
    vector3& v1 = vertices[1];
    vector3& v2 = vertices[2];

    // Shortcuts for UVs
    vector2& uv0 = uvs[0];
    vector2& uv1 = uvs[1];
    vector2& uv2 = uvs[2];

    // Edges of the triangle : position delta
    vector3 deltaPos1 = v1 - v0;
    vector3 deltaPos2 = v2 - v0;

    // UV delta
    vector2 deltaUV1 = uv1 - uv0;
    vector2 deltaUV2 = uv2 - uv0;

    float r = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
    vector3 tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
    vector3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;

    // Set the same tangent for all three vertices of the triangle.
    tangents[0] = tangent;
    tangents[1] = tangent;
    tangents[2] = tangent;

    bitangents[0] = bitangent;
    bitangents[1] = bitangent;
    bitangents[2] = bitangent;
}

__host__ __device__ color mesh_loader::get_color(tinyobj::real_t* raws)
{
    return color(raws[0], raws[1], raws[2]);
}

__host__ __device__ material* mesh_loader::get_mtl_mat(const tinyobj::material_t& reader_mat)
{
    color ambient(0.0, 0.0, 0.0);
    texture* diffuse_a = nullptr;
    texture* specular_a = nullptr;
    texture* bump_a = nullptr;
    texture* normal_a = nullptr;
    texture* displace_a = nullptr;
    texture* alpha_a = nullptr;
    texture* emissive_a = nullptr;
    texture* transparency_a = new solid_color_texture(get_color((tinyobj::real_t*)reader_mat.transmittance) * (1.0f - reader_mat.dissolve));
    texture* sharpness_a = new solid_color_texture(color(1, 0, 0) * reader_mat.shininess);

    float shininess = reader_mat.shininess;

    // Handle textures if necessary (replace std::string usage if needed)

    return new phong(diffuse_a, specular_a, bump_a, normal_a, displace_a, alpha_a, emissive_a, ambient, shininess);
}

__host__ __device__ void mesh_loader::applyDisplacement(mesh_data& data, displacement_texture* tex)
{
    // Replace std::map with a bool array
    size_t num_vertices = data.attributes.vertices.size() / 3;
    bool* processed = new bool[num_vertices];
    memset(processed, 0, num_vertices * sizeof(bool));

    for (size_t s = 0; s < data.num_shapes; ++s)
    {
        tinyobj::shape_t& shape = data.shapes[s];
        size_t num_indices = shape.mesh.indices.size();

        for (size_t i = 0; i < num_indices; i++)
        {
            tinyobj::index_t& idx = shape.mesh.indices[i];

            int vertex_index = idx.vertex_index;

            if (!processed[vertex_index])
            {
                processed[vertex_index] = true;

                float vx = data.attributes.vertices[3 * vertex_index + 0];
                float vy = data.attributes.vertices[3 * vertex_index + 1];
                float vz = data.attributes.vertices[3 * vertex_index + 2];

                float tx = data.attributes.texcoords[2 * idx.texcoord_index + 0];
                float ty = data.attributes.texcoords[2 * idx.texcoord_index + 1];

                color displacement = tex->value(tx, ty, point3());

                vx += vx * (float)displacement.r();
                vy += vy * (float)displacement.g();
                vz += vz * (float)displacement.b();

                data.attributes.vertices[3 * vertex_index + 0] = vx;
                data.attributes.vertices[3 * vertex_index + 1] = vy;
                data.attributes.vertices[3 * vertex_index + 2] = vz;
            }
        }
    }

    delete[] processed;
}
