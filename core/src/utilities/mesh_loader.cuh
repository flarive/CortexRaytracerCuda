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

#include <array>
#include <filesystem>


class mesh_loader
{
public:

    mesh_loader()
    {
    }


    typedef struct
    {
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        tinyobj::attrib_t attributes;
    } mesh_data;


    /// <summary>
    /// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/rtw_stb_obj_loader.h
    /// </summary>
    /// <param name="filepath"></param>
    /// <param name="model_material"></param>
    /// <param name="use_mtl"></param>
    /// <param name="shade_smooth"></param>
    /// <returns></returns>
    static bool load_model_from_file(std::string filepath, mesh_data& mesh);

    static hittable* convert_model_from_file(mesh_data& data, material* model_material, bool use_mtl, bool shade_smooth, std::string name = "");

    static color get_color(tinyobj::real_t* raws);

    static material* get_mtl_mat(const tinyobj::material_t& reader_mat);

    static void computeTangentBasis(std::array<vector3, 3>& vertices, std::array<vector2, 3>& uvs, std::array<vector3, 3>& normals, std::array<vector3, 3>& tangents, std::array<vector3, 3>& bitangents);

    static void applyDisplacement(mesh_data& data, displacement_texture* tex);
};




/// <summary>
/// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/rtw_stb_obj_loader.h
/// </summary>
/// <param name="filename"></param>
/// <param name="model_material"></param>
/// <param name="use_mtl"></param>
/// <param name="shade_smooth"></param>
inline bool mesh_loader::load_model_from_file(std::string filepath, mesh_data& mesh)
{
    // from https://github.com/mojobojo/OBJLoader/blob/master/example.cc
    std::filesystem::path dir(std::filesystem::current_path());
    std::filesystem::path file(filepath);
    std::filesystem::path fullexternalProgramPath = dir / file;

    std::cout << "[INFO] Loading obj file " << fullexternalProgramPath.filename() << std::endl;

    auto fullAbsPath = std::filesystem::absolute(fullexternalProgramPath);

    if (!std::filesystem::exists(fullAbsPath))
    {
        std::cout << "[ERROR] obj file not found ! " << fullAbsPath.generic_string() << std::endl;
        return false;
    }


    std::string inputfile = fullAbsPath.generic_string();
    // By default searches for mtl file in same dir as obj file, and triangulates
    tinyobj::ObjReaderConfig reader_config;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config))
    {
        if (!reader.Error().empty())
        {
            std::cerr << "[ERROR] Loading obj file error: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty())
    {
        std::cerr << "[WARN] Loading obj file warning: " << reader.Warning();
    }

    try
    {
        mesh.attributes = reader.GetAttrib();
        mesh.shapes = reader.GetShapes();
        mesh.materials = reader.GetMaterials();

        return true;
    }
    catch (const std::exception&)
    {
        std::cerr << "[ERROR] Loading obj file failed: " << reader.Error() << std::endl;
        exit(1);
    }

    return false;
}


inline hittable* mesh_loader::convert_model_from_file(mesh_data& data, material* model_material, bool use_mtl, bool shade_smooth, std::string name)
{
    hittable_list model_output;


    int seed = 789999;
    thrust::minstd_rand rng(seed);
    //thrust::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);




    std::cout << "[INFO] Start building obj file (" << data.shapes.size() << " objects found)" << std::endl;

    std::vector<material*> converted_mats;
    for (auto& raw_mat : data.materials)
    {
        converted_mats.push_back(get_mtl_mat(raw_mat));
    }

    const bool use_mtl_file = use_mtl && (data.materials.size() != 0);

    // Loop over shapes
    for (size_t s = 0; s < data.shapes.size(); s++)
    {
        if (use_mtl_file)
        {
            std::string filepath = data.materials[0].displacement_texname;
            double strength = data.materials[0].displacement_texopt.bump_multiplier;

            if (!filepath.empty())
            {
                /*image_texture* image_tex = new image_texture(filepath);
                displacement_texture* displace_texture = new displacement_texture(image_tex, strength);
                if (displace_texture)
                {
                    mesh_loader::applyDisplacement(data, displace_texture);
                }*/
            }
        }
        else if (model_material && model_material->has_displace_texture())
        {
            if (model_material->get_displacement_texture()->getTypeID() == TextureTypeID::textureDisplacementType)
            {
                displacement_texture* displace_texture = static_cast<displacement_texture*>(model_material->get_displacement_texture());
                if (displace_texture)
                {
                    mesh_loader::applyDisplacement(data, displace_texture);
                }
            }
        }

        hittable_list shape_triangles;

        size_t index_offset = 0;

        // Loop over faces(polygon)
        for (size_t f = 0; f < data.shapes[s].mesh.num_face_vertices.size(); f++)
        {
            const int fv = 3;

            assert(data.shapes[s].mesh.num_face_vertices[f] == fv);

            std::array<vector3, 3> tri_v{};
            std::array<vector3, 3> tri_vn{};
            std::array<vector2, 3> tri_uv{};

            // Loop over vertices in the face.
            for (size_t v = 0; v < 3; v++)
            {
                // access to vertex
                tinyobj::index_t idx = data.shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = data.attributes.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = data.attributes.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = data.attributes.vertices[3 * size_t(idx.vertex_index) + 2];

                tri_v[v] = vector3(vx, vy, vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0)
                {
                    tinyobj::real_t nx = data.attributes.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = data.attributes.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = data.attributes.normals[3 * size_t(idx.normal_index) + 2];

                    tri_vn[v] = vector3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0)
                {
                    tinyobj::real_t tu = data.attributes.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t tv = data.attributes.texcoords[2 * size_t(idx.texcoord_index) + 1];

                    tri_uv[v] = vector2(tu, tv);
                }
            }

            // Calculate tangent and bitangent for normal texture
            std::array<vector3, 3> tri_tan{}; // triangle tangents
            std::array<vector3, 3> tri_bitan{}; // triangle bitangents
            computeTangentBasis(tri_v, tri_uv, tri_vn, tri_tan, tri_bitan);

            material* tri_mat;
            if (use_mtl_file)
            {
                tri_mat = converted_mats[data.shapes[s].mesh.material_ids[f]];
            }
            else
            {
                tri_mat = model_material;
            }

            shape_triangles.add(new triangle(
                tri_v[0], tri_v[1], tri_v[2],
                tri_vn[0], tri_vn[1], tri_vn[2],
                tri_uv[0], tri_uv[1], tri_uv[2],
                tri_tan[0], tri_tan[1], tri_tan[2],
                tri_bitan[0], tri_bitan[1], tri_bitan[2],
                shade_smooth, tri_mat));

            index_offset += fv;
        }

        std::cout << "[INFO] Parsing obj file (object name " << data.shapes[s].name << " / " << static_cast<int>(data.attributes.vertices.size() / 3) << " vertex / " << data.shapes[s].mesh.num_face_vertices.size() << " faces)" << std::endl;

        


        // group all object triangles in a bvh node
        //model_output.add(std::make_shared<bvh_node>(shape_triangles, 0, 1));
        model_output.add(new bvh_node(shape_triangles.objects, 0, shape_triangles.object_count, rng, name.c_str()));
    }

    std::cout << "[INFO] End building obj file" << std::endl;


    // group all objects in the .obj file in a single bvh node
    //return std::make_shared<bvh_node>(model_output, 0, 1);
    return new bvh_node(model_output.objects, 0, model_output.object_count, rng, name.c_str());
}

/// <summary>
/// https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/#computing-the-tangents-and-bitangents
/// </summary>
/// <param name="vertices"></param>
/// <param name="uvs"></param>
/// <param name="normals"></param>
/// <param name="tangents"></param>
/// <param name="bitangents"></param>
inline void mesh_loader::computeTangentBasis(std::array<vector3, 3>& vertices, std::array<vector2, 3>& uvs, std::array<vector3, 3>& normals, std::array<vector3, 3>& tangents, std::array<vector3, 3>& bitangents)
{
    //For each triangle, we compute the edge(deltaPos) and the deltaUV
    for (uint64_t i = 0; i < vertices.size(); i += 3)
    {
        // Shortcuts for vertices
        vector3& v0 = vertices[i];
        vector3& v1 = vertices[i + 1];
        vector3& v2 = vertices[i + 2];

        // Shortcuts for UVs
        vector2& uv0 = uvs[i];
        vector2& uv1 = uvs[i + 1];
        vector2& uv2 = uvs[i + 2];

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
        // They will be merged later, in vboindexer.cpp
        tangents[0] = tangent;
        tangents[1] = tangent;
        tangents[2] = tangent;

        // Same thing for bitangents
        bitangents[0] = bitangent;
        bitangents[1] = bitangent;
        bitangents[2] = bitangent;
    }
}

inline color mesh_loader::get_color(tinyobj::real_t* raws)
{
    return color(raws[0], raws[1], raws[2]);
}


inline material* mesh_loader::get_mtl_mat(const tinyobj::material_t& reader_mat)
{
    color ambient(0.0, 0.0, 0.0);
    texture* diffuse_a = nullptr;
    texture* specular_a = nullptr;
    texture* bump_a = nullptr;
    texture* normal_a = nullptr;
    texture* displace_a = nullptr;
    texture* alpha_a = nullptr;
    //texture* emissive_a = new solid_color_texture(get_color((tinyobj::real_t*)reader_mat.emission));
    texture* emissive_a = nullptr;
    texture* transparency_a = new solid_color_texture(get_color((tinyobj::real_t*)reader_mat.transmittance) * (1.0f - reader_mat.dissolve));
    texture* sharpness_a = new solid_color_texture(color(1, 0, 0) * reader_mat.shininess);

    // Ns
    float shininess = reader_mat.shininess; // 0.0

    // diffuse
    // map_Kd ..\..\data\models\crate_diffuse.jpg
    //if (!reader_mat.diffuse_texname.empty())
    //{
    //    diffuse_a = new image_texture(reader_mat.diffuse_texname);
    //}
    //else
    //{
    //    diffuse_a = new solid_color_texture(get_color((tinyobj::real_t*)reader_mat.diffuse));
    //}


    // specular
    // map_Ks ..\..\data\models\crate_roughness.jpg
    //if (!reader_mat.specular_texname.empty())
    //{
    //    specular_a = new image_texture(reader_mat.specular_texname);
    //}
    //else
    //{
    //    specular_a = new solid_color_texture(get_color((tinyobj::real_t*)reader_mat.specular));
    //}

    // bump
    // map_bump -bm 0.3000 ..\..\data\models\crate_bump.jpg
    //if (!reader_mat.bump_texname.empty())
    //{
    //    // bump strength
    //    auto bump_m = reader_mat.bump_texopt.bump_multiplier;

    //    // bump texture
    //    auto bump_tex = new image_texture(reader_mat.bump_texname);
    //    bump_a = new bump_texture(bump_tex, bump_m);
    //}

    //// normal
    //// norm -bm 0.3000 ..\..\data\models\crate_normal.jpg
    //if (!reader_mat.normal_texname.empty())
    //{
    //    // normal strength
    //    double normal_m = reader_mat.normal_texopt.bump_multiplier;

    //    // normal texture
    //    auto normal_tex = new image_texture(reader_mat.normal_texname);
    //    normal_a = new normal_texture(normal_tex, normal_m);
    //}

    //// emissive
    //// 
    //if (!reader_mat.emissive_texname.empty())
    //{

    //}

    //// displacement/height
    //// disp -bm 1.0 ..\..\data\models\rocky_normal.jpg
    //if (!reader_mat.displacement_texname.empty())
    //{
    //    // displace strength
    //    double displace_m = (double)reader_mat.displacement_texopt.bump_multiplier;

    //    // displace texture
    //    auto displace_tex = new image_texture(reader_mat.displacement_texname);
    //    displace_a = new displacement_texture(displace_tex, displace_m);
    //}

    return new phong(diffuse_a, specular_a, bump_a, normal_a, displace_a, alpha_a, emissive_a, ambient, shininess);
}

inline void mesh_loader::applyDisplacement(mesh_data& data, displacement_texture* tex)
{
    //std::cout << "[INFO] Start applying model displacement " << data.shapes.size() << std::endl;

    // temp dic to take each vertex only one time
    std::map<int, bool> dic;

    for (auto& shape : data.shapes)
    {
        for (size_t i = 0; i < shape.mesh.indices.size(); i++)
        {
            auto& idx = shape.mesh.indices[i];

            if (dic.find(idx.vertex_index) == dic.end())
            {
                // dic does not contain vertex yet
                dic.emplace(idx.vertex_index, true);

                float vx = data.attributes.vertices[3 * idx.vertex_index + 0];
                float vy = data.attributes.vertices[3 * idx.vertex_index + 1];
                float vz = data.attributes.vertices[3 * idx.vertex_index + 2];

                float tx = data.attributes.texcoords[2 * idx.texcoord_index + 0];
                float ty = data.attributes.texcoords[2 * idx.texcoord_index + 1];


                color displacement = tex->value(tx, ty, point3());


                vx += vx * (float)displacement.r();
                vy += vy * (float)displacement.g();
                vz += vz * (float)displacement.b();

                data.attributes.vertices[3 * idx.vertex_index + 0] = vx;
                data.attributes.vertices[3 * idx.vertex_index + 1] = vy;
                data.attributes.vertices[3 * idx.vertex_index + 2] = vz;
            }
        }
    }

    //std::cout << "[INFO] End applying model displacement" << std::endl;
}