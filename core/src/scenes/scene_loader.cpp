#include "scene_loader.h"

#include "../utilities/uvmapping.cuh"
#include "../misc/transform.cuh"

#include "iostream"
#include <filesystem>

//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>




scene_loader::scene_loader(std::string path) : _path(std::move(path))
{
}

scene_builder scene_loader::loadSceneFromFile()
{
	scene_builder builder;

	if (_path.empty())
	{
		std::cerr << "[ERROR] No scene to load !" << std::endl;
	}

	std::filesystem::path dir(std::filesystem::current_path());
	std::filesystem::path file(_path.c_str());
	std::filesystem::path fullexternalProgramPath = dir / file;

	std::cout << "[INFO] Loading scene " << fullexternalProgramPath.filename() << std::endl;

	auto fullAbsPath = std::filesystem::absolute(fullexternalProgramPath);

	if (std::filesystem::exists(fullAbsPath))
	{
		try
		{
			this->cfg.readFile(fullAbsPath.string());
		}
		catch (const libconfig::ParseException& e)
		{
			std::cerr << "[ERROR] Scene parsing error line " << e.getLine() << " " << e.getError() <<  std::endl;
			//std::cerr << "Error : " << e.getError() << std::endl;
			//std::cerr << "Line : " << e.getLine() << std::endl;
			//std::cerr << "File : " << e.getFile() << std::endl;
			//std::cerr << "Press any key to quit..." << std::endl;
			system("pause");
			return builder;
		}
		catch (const std::exception& e)
		{
			std::cerr << "[ERROR] Scene loading failed ! " << e.what() << std::endl;
			//std::cerr << "Press any key to quit..." << std::endl;
			system("pause");
			return builder;
		}
		
		std::cout << "[INFO] Building scene" << std::endl;

		const libconfig::Setting& root = this->cfg.getRoot();

		if (root.exists("image"))
		{
			const libconfig::Setting& image = root["image"];
			this->loadImageConfig(builder, image);
		}

		if (root.exists("camera"))
		{
			const libconfig::Setting& camera = root["camera"];
			this->loadCameraConfig(builder, camera);
		}

		if (root.exists("lights"))
		{
			const libconfig::Setting& lights = root["lights"];
			this->loadLights(builder, lights);
		}

		if (root.exists("textures"))
		{
			const libconfig::Setting& textures = root["textures"];
			this->loadTextures(builder, textures);
		}

		if (root.exists("materials"))
		{
			const libconfig::Setting& materials = root["materials"];
			this->loadMaterials(builder, materials);
		}

		if (root.exists("primitives"))
		{
			const libconfig::Setting& primitives = root["primitives"];
			this->loadPrimitives(builder, primitives);
		}

		if (root.exists("meshes"))
		{
			const libconfig::Setting& meshes = root["meshes"];
			this->loadMeshes(builder, meshes);
		}

		if (root.exists("groups"))
		{
			const libconfig::Setting& groups = root["groups"];
			this->loadGroups(builder, groups);
		}
	}
	else
	{
		std::cerr << "[ERROR] Scene not found ! " << fullAbsPath << std::endl;
	}

	return builder;
}

void scene_loader::loadPrimitives(scene_builder& builder, const libconfig::Setting& primitives)
{
	addSpherePrimitives(primitives, builder);
	addPlanePrimitives(primitives, builder);
	addQuadPrimitives(primitives, builder);
	addBoxPrimitives(primitives, builder);
	addConePrimitives(primitives, builder);
	addCylinderPrimitives(primitives, builder);
	addDiskPrimitives(primitives, builder);
	addTorusPrimitives(primitives, builder);
	addVolumePrimitives(primitives, builder);
}

void scene_loader::loadMaterials(scene_builder& builder, const libconfig::Setting& materials)
{
	uint8_t countLambertianMaterials = 0;
	uint8_t countMetalMaterials = 0;
	uint8_t countDielectricMaterials = 0;
	uint8_t countIsotropicMaterials = 0;
	uint8_t countAnisotropicMaterials = 0;
	uint8_t countOrenNayarMaterials = 0;
	uint8_t countPhongMaterials = 0;

	if (materials.exists("lambertian"))
		countLambertianMaterials = materials["lambertian"].getLength();

	if (materials.exists("metal"))
		countMetalMaterials = materials["metal"].getLength();

	if (materials.exists("dielectric"))
		countDielectricMaterials = materials["dielectric"].getLength();

	if (materials.exists("isotropic"))
		countIsotropicMaterials = materials["isotropic"].getLength();

	if (materials.exists("anisotropic"))
		countAnisotropicMaterials = materials["anisotropic"].getLength();

	if (materials.exists("orennayar"))
		countOrenNayarMaterials = materials["orennayar"].getLength();

	if (materials.exists("phong"))
		countPhongMaterials = materials["phong"].getLength();


	builder.initMaterialsConfig(countLambertianMaterials, countMetalMaterials, countDielectricMaterials, countIsotropicMaterials, countAnisotropicMaterials, countOrenNayarMaterials, countPhongMaterials);

	addLambertianMaterial(materials, builder);
	addPhongMaterial(materials, builder);
	addOrenNayarMaterial(materials, builder);
	addIsotropicMaterial(materials, builder);
	addAnisotropicMaterial(materials, builder);
	addGlassMaterial(materials, builder);
	addMetalMaterial(materials, builder);
}

void scene_loader::loadTextures(scene_builder& builder, const libconfig::Setting& textures)
{
	uint8_t countSolidColorTextures = 0;
	uint8_t countGradientColorTextures = 0;
	uint8_t countImageTextures = 0;
	uint8_t countNoiseTextures = 0;
	uint8_t countCheckerTextures = 0;
	uint8_t countBumpTextures = 0;
	uint8_t countNormalTextures = 0;
	uint8_t countDisplacementTextures = 0;
	uint8_t countAlphaTextures = 0;
	uint8_t countEmissiveTextures = 0;

	if (textures.exists("solidColor"))
		countSolidColorTextures = textures["solidColor"].getLength();

	if (textures.exists("gradientColor"))
		countGradientColorTextures = textures["gradientColor"].getLength();

	if (textures.exists("image"))
		countImageTextures = textures["image"].getLength();

	if (textures.exists("noise"))
		countNoiseTextures = textures["noise"].getLength();

	if (textures.exists("checker"))
		countCheckerTextures = textures["checker"].getLength();

	if (textures.exists("bump"))
		countBumpTextures = textures["bump"].getLength();

	if (textures.exists("normal"))
		countNormalTextures = textures["normal"].getLength();

	if (textures.exists("displacement"))
		countDisplacementTextures = textures["displacement"].getLength();

	if (textures.exists("alpha"))
		countAlphaTextures = textures["alpha"].getLength();

	if (textures.exists("emissive"))
		countEmissiveTextures = textures["emissive"].getLength();


	builder.initTexturesConfig(countSolidColorTextures, countGradientColorTextures, countImageTextures, countNoiseTextures, countCheckerTextures, countBumpTextures, countNormalTextures, countDisplacementTextures, countAlphaTextures, countEmissiveTextures);

	addSolidColorTexture(textures, builder);
	addGradientColorTexture(textures, builder);
	addImageTexture(textures, builder);
	addNoiseTexture(textures, builder);
	addCheckerTexture(textures, builder);
	addBumpTexture(textures, builder);
	addNormalTexture(textures, builder);
	addDisplacementTexture(textures, builder);
	addAlphaTexture(textures, builder);
	addEmissiveTexture(textures, builder);
}



void scene_loader::loadLights(scene_builder& builder, const libconfig::Setting& lights)
{
		uint8_t countOmniLights = 0;
		uint8_t countDirLights = 0;
		uint8_t countSpotLights = 0;
		
		if (lights.exists("omnis"))
			countOmniLights = lights["omnis"].getLength();

		if (lights.exists("directionals"))
			countDirLights = lights["directionals"].getLength();

		if (lights.exists("spots"))
			countSpotLights = lights["spots"].getLength();


		builder.initLightsConfig(countOmniLights, countDirLights, countSpotLights);

		addDirectionalLight(lights, builder);
		addOmniLight(lights, builder);
		addSpotLight(lights, builder);
}

void scene_loader::loadImageConfig(scene_builder& builder, const libconfig::Setting& setting)
{
	if (setting.exists("width"))
		builder.imageWidth(setting["width"]);
	if (setting.exists("height"))
		builder.imageHeight(setting["height"]);
	if (setting.exists("maxDepth"))
		builder.imageDepth(setting["maxDepth"]);
	if (setting.exists("samplesPerPixel"))
		builder.imageSamplesPerPixel(setting["samplesPerPixel"]);
	if (setting.exists("background"))
		loadImageBackgroundConfig(builder, setting["background"]);
	if (setting.exists("outputFilePath"))
		builder.imageOutputFilePath(setting["outputFilePath"].c_str());
}

void scene_loader::loadImageBackgroundConfig(scene_builder& builder, const libconfig::Setting& setting)
{
	color rgb{};
	std::string filepath;
	bool is_skybox = false;

	if (setting.exists("color"))
		rgb = getColor(setting["color"]);

	if (setting.exists("filepath"))
		setting.lookupValue("filepath", filepath);

	if (setting.exists("is_skybox"))
		setting.lookupValue("is_skybox", is_skybox);
	
	builder.setImageBackgroundConfig(rgb, filepath.c_str(), is_skybox);
}

void scene_loader::loadCameraConfig(scene_builder& builder, const libconfig::Setting& setting)
{
	if (setting.exists("aspectRatio")) {
		builder.cameraAspectRatio(setting["aspectRatio"]);
	}
	if (setting.exists("openingTime")) {
		builder.cameraOpeningTime(setting["openingTime"]);
	}
	if (setting.exists("lookFrom")) {
		builder.cameraLookFrom(this->getPoint(setting["lookFrom"]));
	}
	if (setting.exists("lookAt")) {
		builder.cameraLookAt(this->getPoint(setting["lookAt"]));
	}
	if (setting.exists("upAxis")) {
		builder.cameraUpAxis(this->getPoint(setting["upAxis"]));
	}
	if (setting.exists("aperture")) {
		builder.cameraAperture(setting["aperture"]);
	}
	if (setting.exists("focus")) {
		builder.cameraFocus(setting["focus"]);
	}
	if (setting.exists("fov")) {
		builder.cameraFOV(setting["fov"]);
	}
	if (setting.exists("orthographic")) {
		builder.cameraIsOrthographic(setting["orthographic"]);
	}
	if (setting.exists("orthoHeight")) {
		builder.cameraOrthoHeight(setting["orthoHeight"]);
	}
}

void scene_loader::loadMeshes(scene_builder& builder, const libconfig::Setting& setting)
{
	if (setting.exists("obj"))
	{
		for (int i = 0; i < setting["obj"].getLength(); i++)
		{
			const libconfig::Setting& mesh = setting["obj"][i];
			std::string name;
			std::string filePath;
			point3 position{};
			std::string materialName;
			bool use_mtl = true;
			bool use_smoothing = true;
			std::string groupName;
			bool active = true;

			if (mesh.exists("name"))
				mesh.lookupValue("name", name);
			if (mesh.exists("position"))
				position = this->getPoint(mesh["position"]);
			if (mesh.exists("filepath"))
				mesh.lookupValue("filepath", filePath);
			if (mesh.exists("material"))
				mesh.lookupValue("material", materialName);
			if (mesh.exists("use_mtl"))
				mesh.lookupValue("use_mtl", use_mtl);
			if (mesh.exists("use_smoothing"))
				mesh.lookupValue("use_smoothing", use_smoothing);
			if (mesh.exists("group"))
				mesh.lookupValue("group", groupName);
			if (mesh.exists("active"))
				mesh.lookupValue("active", active);

			if (active)
			{
				builder.addMesh(name.c_str(), position, filePath.c_str(), materialName.c_str(), use_mtl, use_smoothing, groupName.c_str());

				applyTransform(mesh, builder, name.c_str());
			}
		}
	}
}

void scene_loader::loadGroups(scene_builder& builder, const libconfig::Setting& setting)
{
	for (int i = 0; i < setting.getLength(); i++)
	{
		const libconfig::Setting& group = setting[i];
		std::string name;

		if (group.exists("name"))
			group.lookupValue("name", name);

		bool groupIsUsed = false;
		builder.addGroup(name.c_str(), groupIsUsed);

		if (groupIsUsed)
			applyTransform(group, builder, name.c_str());
	}
}
	
void scene_loader::applyTransform(const libconfig::Setting& primitive, scene_builder& builder, const char* name)
{
	if (primitive.exists("transform"))
	{
		rt::transform transform = this->getTransform(primitive["transform"]);

		if (transform.hasTranslate())
			builder.translate(transform.getTranslate(), name);
		if (transform.hasRotate())
			builder.rotate(transform.getRotate(), name);
		if (transform.hasScale())
			builder.scale(transform.getScale(), name);
	}
}

void scene_loader::addImageTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("image"))
	{
		const libconfig::Setting& image = textures["image"];

		for (int i = 0; i < image.getLength(); i++)
		{
			const libconfig::Setting& texture = image[i];
			std::string name;
			std::string filepath;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);

			if (name.empty())
				throw std::runtime_error("Image texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addImageTexture(name.c_str(), filepath.c_str());
			}
		}
	}
}

void scene_loader::addNoiseTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("noise"))
	{
		const libconfig::Setting& noise = textures["noise"];

		for (int i = 0; i < noise.getLength(); i++)
		{
			const libconfig::Setting& texture = noise[i];
			std::string name;
			float scale = 1.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("scale"))
				texture.lookupValue("scale", scale);

			if (name.empty())
				throw std::runtime_error("Noise texture name is empty");

			builder.addNoiseTexture(name.c_str(), scale);
		}
	}
}

void scene_loader::addSolidColorTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("solidColor"))
	{
		const libconfig::Setting& texs = textures["solidColor"];

		for (int i = 0; i < texs.getLength(); i++)
		{
			const libconfig::Setting& texture = texs[i];
			std::string name;
			color color = { 0.0, 0.0, 0.0 };

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("color"))
				color = this->getColor(texture["color"]);

			if (name.empty())
				throw std::runtime_error("Solid color texture name is empty");

			builder.addSolidColorTexture(name.c_str(), color);
		}
	}
}

void scene_loader::addCheckerTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("checker"))
	{
		const libconfig::Setting& texs = textures["checker"];

		for (int i = 0; i < texs.getLength(); i++)
		{
			const libconfig::Setting& texture = texs[i];
			std::string name;
			std::string oddTextureName;
			std::string evenTextureName;
			color oddColor{};
			color evenColor{};
			float scale = 0.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("scale"))
				texture.lookupValue("scale", scale);
			if (texture.exists("oddTextureName"))
				texture.lookupValue("oddTextureName", oddTextureName);
			if (texture.exists("evenTextureName"))
				texture.lookupValue("evenTextureName", evenTextureName);
			if (texture.exists("oddColor"))
				oddColor = this->getColor(texture["oddColor"]);
			if (texture.exists("evenColor"))
				evenColor = this->getColor(texture["evenColor"]);

			if (name.empty())
				throw std::runtime_error("Checker texture name is empty");

			builder.addCheckerTexture(name.c_str(), scale, oddTextureName.c_str(), evenTextureName.c_str(), oddColor, evenColor);
;		}
	}
}

void scene_loader::addGradientColorTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("gradientColor"))
	{
		const libconfig::Setting& texs = textures["gradientColor"];

		for (int i = 0; i < texs.getLength(); i++)
		{
			const libconfig::Setting& texture = texs[i];
			std::string name;
			color color1 = { 0.0, 0.0, 0.0 };
			color color2 = { 1.0, 1.0, 1.0 };
			bool vertical = true;
			bool hsv = false;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("color1"))
				color1 = this->getColor(texture["color1"]);
			if (texture.exists("color2"))
				color2 = this->getColor(texture["color2"]);
			if (texture.exists("vertical"))
				texture.lookupValue("vertical", vertical);
			if (texture.exists("hsv"))
				texture.lookupValue("hsv", hsv);

			if (name.empty())
				throw std::runtime_error("Gradient color texture name is empty");

			builder.addGradientColorTexture(name.c_str(), color1, color2, !vertical, hsv);
		}
	}
}

void scene_loader::addBumpTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("bump"))
	{
		const libconfig::Setting& texs = textures["bump"];

		for (int i = 0; i < texs.getLength(); i++)
		{
			const libconfig::Setting& texture = texs[i];
			std::string name;
			std::string filepath;
			float strength = 0.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);
			if (texture.exists("strength"))
				texture.lookupValue("strength", strength);

			if (name.empty())
				throw std::runtime_error("Bump texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addBumpTexture(name.c_str(), filepath.c_str(), strength);
			}
		}
	}
}

void scene_loader::addNormalTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("normal"))
	{
		const libconfig::Setting& image = textures["normal"];

		for (int i = 0; i < image.getLength(); i++)
		{
			const libconfig::Setting& texture = image[i];
			std::string name;
			std::string filepath;
			float strength = 10.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);
			if (texture.exists("strength"))
				texture.lookupValue("strength", strength);

			if (name.empty())
				throw std::runtime_error("Normal texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addNormalTexture(name.c_str(), filepath.c_str(), strength);
			}
		}
	}
}

void scene_loader::addDisplacementTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("displacement"))
	{
		const libconfig::Setting& image = textures["displacement"];

		for (int i = 0; i < image.getLength(); i++)
		{
			const libconfig::Setting& texture = image[i];
			std::string name;
			std::string filepath;
			float strength = 10.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);
			if (texture.exists("strength"))
				texture.lookupValue("strength", strength);

			if (name.empty())
				throw std::runtime_error("Displacement texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addDisplacementTexture(name.c_str(), filepath.c_str(), strength);
			}
		}
	}
}

void scene_loader::addAlphaTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("alpha"))
	{
		const libconfig::Setting& image = textures["alpha"];

		for (int i = 0; i < image.getLength(); i++)
		{
			const libconfig::Setting& texture = image[i];
			std::string name;
			std::string filepath;
			bool double_sided = false;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);
			if (texture.exists("double_sided"))
				texture.lookupValue("double_sided", double_sided);

			if (name.empty())
				throw std::runtime_error("Alpha texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addAlphaTexture(name.c_str(), filepath.c_str(), double_sided);
			}
		}
	}
}

void scene_loader::addEmissiveTexture(const libconfig::Setting& textures, scene_builder& builder)
{
	if (textures.exists("emissive"))
	{
		const libconfig::Setting& image = textures["emissive"];

		for (int i = 0; i < image.getLength(); i++)
		{
			const libconfig::Setting& texture = image[i];
			std::string name;
			std::string filepath;
			float strength = 10.0f;

			if (texture.exists("name"))
				texture.lookupValue("name", name);
			if (texture.exists("filepath"))
				texture.lookupValue("filepath", filepath);
			if (texture.exists("strength"))
				texture.lookupValue("strength", strength);

			if (name.empty())
				throw std::runtime_error("Emissive texture name is empty");

			if (!filepath.empty())
			{
				//int bytes_per_pixel = 3;
				//int tex_x, tex_y, tex_n;
				//unsigned char* tex_data_host = stbi_load(filepath.c_str(), &tex_x, &tex_y, &tex_n, bytes_per_pixel);
				//if (!tex_data_host) {
				//	std::cerr << "[ERROR] Failed to load texture." << std::endl;
				//	return;
				//}

				//bitmap_image img = bitmap_image(tex_data_host, tex_x, tex_y, bytes_per_pixel);
				builder.addEmissiveTexture(name.c_str(), filepath.c_str(), strength);
			}
		}
	}
}

void scene_loader::addDirectionalLight(const libconfig::Setting& lights, scene_builder& builder)
{
	if (lights.exists("directionals"))
	{
		for (int i = 0; i < lights["directionals"].getLength(); i++)
		{
			const libconfig::Setting& light = lights["directionals"][i];
			std::string name{};
			color rgb{};
			point3 position{};
			vector3 u{};
			vector3 v{};
			float intensity = 0.0f;
			bool invisible = true;
			bool active = true;

			if (light.exists("name"))
				light.lookupValue("name", name);
			if (light.exists("position"))
				position = this->getVector(light["position"]);
			if (light.exists("u"))
				u = this->getPoint(light["u"]);
			if (light.exists("v"))
				v = this->getPoint(light["v"]);
			if (light.exists("intensity"))
				light.lookupValue("intensity", intensity);
			if (light.exists("color"))
				rgb = this->getColor(light["color"]);
			if (light.exists("invisible"))
				light.lookupValue("invisible", invisible);
			if (light.exists("active"))
				light.lookupValue("active", active);

			if (active)
			{
				builder.addDirectionalLight(position, u, v, intensity, rgb, invisible, const_cast<char*>(name.c_str()));

				//applyTransform(light, builder, name.c_str());


			}
		}
	}
}

void scene_loader::addOmniLight(const libconfig::Setting& lights, scene_builder& builder)
{
	if (lights.exists("omnis"))
	{
		for (int i = 0; i < lights["omnis"].getLength(); i++)
		{
			const libconfig::Setting& light = lights["omnis"][i];
			std::string name{};
			color rgb{};
			point3 position{};
			float radius = 0.0f;
			float intensity = 0.0f;
			bool invisible = true;
			bool active = true;

			if (light.exists("name"))
				light.lookupValue("name", name);
			if (light.exists("position"))
				position = this->getVector(light["position"]);
			if (light.exists("radius"))
				light.lookupValue("radius", radius);
			if (light.exists("intensity"))
				light.lookupValue("intensity", intensity);
			if (light.exists("color"))
				rgb = this->getColor(light["color"]);
			if (light.exists("invisible"))
				light.lookupValue("invisible", invisible);
			if (light.exists("active"))
				light.lookupValue("active", active);

			if (active)
			{
				builder.addOmniDirectionalLight(position, radius, intensity, rgb, invisible, name.c_str());

				//applyTransform(light, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addSpotLight(const libconfig::Setting& lights, scene_builder& builder)
{
	if (lights.exists("spots"))
	{
		for (int i = 0; i < lights["spots"].getLength(); i++)
		{
			const libconfig::Setting& light = lights["spots"][i];
			std::string name{};
			point3 position{};
			vector3 direction{};
			float cutoff = 0.0f;
			float falloff = 0.0f;
			float intensity = 0.0f;
			float radius = 0.0f;
			color rgb{};
			bool invisible = true;
			bool active = true;

			if (light.exists("name"))
				light.lookupValue("name", name);
			if (light.exists("position"))
				position = this->getVector(light["position"]);
			if (light.exists("direction"))
				direction = this->getVector(light["direction"]);
			if (light.exists("cutoff"))
				light.lookupValue("cutoff", cutoff);
			if (light.exists("falloff"))
				light.lookupValue("falloff", falloff);
			if (light.exists("intensity"))
				light.lookupValue("intensity", intensity);
			if (light.exists("radius"))
				light.lookupValue("radius", radius);
			if (light.exists("color"))
				rgb = this->getColor(light["color"]);
			if (light.exists("invisible"))
				light.lookupValue("invisible", invisible);
			if (light.exists("active"))
				light.lookupValue("active", active);

			if (active)
			{
				builder.addSpotLight(position, direction, cutoff, falloff, intensity, radius, rgb, invisible, name.c_str());

				//applyTransform(light, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addLambertianMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("lambertian"))
	{
		for (int i = 0; i < materials["lambertian"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["lambertian"][i];
			std::string name{};
			color rgb{};
			std::string textureName{};

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("color"))
				rgb = this->getColor(material["color"]);
			if (material.exists("texture"))
				material.lookupValue("texture", textureName);

			if (name.empty())
				throw std::runtime_error("Material name is empty");

			builder.addLambertianMaterial(name.c_str(), rgb, textureName.c_str());
		}
	}
}

void scene_loader::addPhongMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("phong"))
	{
		for (int i = 0; i < materials["phong"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["phong"][i];
			std::string name{};
			std::string diffuseTextureName;
			std::string specularTextureName;
			std::string bumpTextureName;
			std::string normalTextureName;
			std::string displacementTextureName;
			std::string alphaTextureName;
			std::string emissiveTextureName;
			color ambientColor{};
			float shininess = 0.0f;

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("diffuseTexture"))
				material.lookupValue("diffuseTexture", diffuseTextureName);
			if (material.exists("specularTexture"))
				material.lookupValue("specularTexture", specularTextureName);
			if (material.exists("bumpTexture"))
				material.lookupValue("bumpTexture", bumpTextureName);
			if (material.exists("normalTexture"))
				material.lookupValue("normalTexture", normalTextureName);
			if (material.exists("displacementTexture"))
				material.lookupValue("displacementTexture", displacementTextureName);
			if (material.exists("alphaTexture"))
				material.lookupValue("alphaTexture", alphaTextureName);
			if (material.exists("emissiveTexture"))
				material.lookupValue("emissiveTexture", emissiveTextureName);
			if (material.exists("ambientColor"))
				ambientColor = this->getColor(material["ambientColor"]);
			if (material.exists("shininess"))
				material.lookupValue("shininess", shininess);

			if (name.empty())
				throw std::runtime_error("Material name is empty");

			if (!diffuseTextureName.empty() || !specularTextureName.empty() || !bumpTextureName.empty() || !normalTextureName.empty() || !displacementTextureName.empty() || !alphaTextureName.empty() || !emissiveTextureName.empty())
				builder.addPhongMaterial(name.c_str(), diffuseTextureName.c_str(), specularTextureName.c_str(), normalTextureName.c_str(), bumpTextureName.c_str(), displacementTextureName.c_str(), alphaTextureName.c_str(), emissiveTextureName.c_str(), ambientColor, shininess);
		}
	}
}

void scene_loader::addOrenNayarMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("orennayar"))
	{
		for (int i = 0; i < materials["orennayar"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["orennayar"][i];
			std::string name{};
			color rgb{};
			std::string textureName{};
			float albedo_temp = 0.0f;
			float roughness = 0.0f;

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("color"))
				rgb = this->getColor(material["color"]);
			if (material.exists("texture"))
				material.lookupValue("texture", textureName);
			if (material.exists("albedo_temp"))
				material.lookupValue("albedo_temp", albedo_temp);
			if (material.exists("roughness"))
				material.lookupValue("roughness", roughness);

			if (name.empty())
				throw std::runtime_error("Material name is empty");

			builder.addOrenNayarMaterial(name.c_str(), rgb, textureName.c_str(), albedo_temp, roughness);
		}
	}
}

void scene_loader::addIsotropicMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("isotropic"))
	{
		for (int i = 0; i < materials["isotropic"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["isotropic"][i];
			std::string name{};
			color rgb{};
			std::string textureName{};

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("color"))
				rgb = this->getColor(material["color"]);
			if (material.exists("texture"))
				material.lookupValue("texture", textureName);

			if (name.empty())
				throw std::runtime_error("Material name is empty");

			if (!textureName.empty())
				builder.addIsotropicMaterial(name.c_str(), textureName.c_str());
			else
				builder.addIsotropicMaterial(name.c_str(), rgb);
		}
	}
}

void scene_loader::addAnisotropicMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("anisotropic"))
	{
		for (int i = 0; i < materials["anisotropic"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["anisotropic"][i];
			std::string name{};
			color rgb{};
			float nu = 0.0f;
			float nv = 0.0f;
			std::string diffuseTextureName;
			std::string specularTextureName;
			std::string exponentTextureName;
			float roughness = 0.0f;

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("color"))
				rgb = this->getColor(material["color"]);
			if (material.exists("nu"))
				material.lookupValue("nu", nu);
			if (material.exists("nv"))
				material.lookupValue("nv", nv);
			if (material.exists("diffuseTextureName"))
				material.lookupValue("diffuseTextureName", diffuseTextureName);
			if (material.exists("specularTextureName"))
				material.lookupValue("specularTextureName", specularTextureName);
			if (material.exists("exponentTextureName"))
				material.lookupValue("exponentTextureName", exponentTextureName);


			if (name.empty())
				throw std::runtime_error("Material name is empty");

			if (!diffuseTextureName.empty())
				builder.addAnisotropicMaterial(name.c_str(), nu, nv, diffuseTextureName.c_str(), specularTextureName.c_str(), exponentTextureName.c_str());
			else
				builder.addAnisotropicMaterial(name.c_str(), nu, nv, rgb);
		}
	}
}

void scene_loader::addGlassMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("glass"))
	{
		for (int i = 0; i < materials["glass"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["glass"][i];
			std::string name;
			float refraction = 0.0f;

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("refraction"))
				material.lookupValue("refraction", refraction);

			if (name.empty())
				throw std::runtime_error("Material name is empty");

			builder.addGlassMaterial(name.c_str(), refraction);
		}
	}
}

void scene_loader::addMetalMaterial(const libconfig::Setting& materials, scene_builder& builder)
{
	if (materials.exists("metal"))
	{
		for (int i = 0; i < materials["metal"].getLength(); i++)
		{
			const libconfig::Setting& material = materials["metal"][i];
			std::string name;
			color color = { 0.0, 0.0, 0.0 };
			float fuzziness = 0.0f;

			if (material.exists("name"))
				material.lookupValue("name", name);
			if (material.exists("color"))
				color = this->getColor(material["color"]);
			if (material.exists("fuzziness"))
				material.lookupValue("fuzziness", fuzziness);
			if (name.empty())
				throw std::runtime_error("Material name is empty");

			builder.addMetalMaterial(name.c_str(), color, fuzziness);
		}
	}
}

void scene_loader::addSpherePrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("spheres"))
	{
		for (int i = 0; i < primitives["spheres"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["spheres"][i];
			std::string name;
			point3 position{};
			float radius = 0.0f;
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("radius"))
				primitive.lookupValue("radius", radius);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addSphere(name.c_str(), position, radius, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addPlanePrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("planes"))
	{
		for (int i = 0; i < primitives["planes"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["planes"][i];
			std::string name;
			point3 point1{};
			point3 point2{};
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("point1"))
				point1 = this->getPoint(primitive["point1"]);
			if (primitive.exists("point2"))
				point2 = this->getPoint(primitive["point2"]);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addPlane(name.c_str(), point1, point2, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addQuadPrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("quads"))
	{
		for (int i = 0; i < primitives["quads"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["quads"][i];
			std::string name;
			point3 position{};
			vector3 u{};
			vector3 v{};
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("u"))
				u = this->getVector(primitive["u"]);
			if (primitive.exists("v"))
				v = this->getVector(primitive["v"]);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addQuad(name.c_str(), position, u, v, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addBoxPrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("boxes"))
	{
		for (int i = 0; i < primitives["boxes"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["boxes"][i];
			std::string name;
			point3 position{};
			point3 size{};
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getVector(primitive["position"]);
			if (primitive.exists("size"))
				size = this->getVector(primitive["size"]);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addBox(name.c_str(), position, size, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addConePrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("cones"))
	{
		for (int i = 0; i < primitives["cones"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["cones"][i];
			std::string name;
			point3 position{};
			float radius = 0.0f;
			float height = 0.0f;
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("radius"))
				primitive.lookupValue("radius", radius);
			if (primitive.exists("height"))
				primitive.lookupValue("height", height);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addCone(name.c_str(), position, radius, height, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addCylinderPrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("cylinders"))
	{
		for (int i = 0; i < primitives["cylinders"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["cylinders"][i];
			std::string name;
			point3 position{};
			float radius = 0.0f;
			float height = 0.0f;
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("radius"))
				primitive.lookupValue("radius", radius);
			if (primitive.exists("height"))
				primitive.lookupValue("height", height);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addCylinder(name.c_str(), position, radius, height, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addDiskPrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("disks"))
	{
		for (int i = 0; i < primitives["disks"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["disks"][i];
			std::string name;
			point3 position{};
			float radius = 0.0f;
			float height = 0.0f;
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("radius"))
				primitive.lookupValue("radius", radius);
			if (primitive.exists("height"))
				primitive.lookupValue("height", height);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addDisk(name.c_str(), position, radius, height, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addTorusPrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("toruses"))
	{
		for (int i = 0; i < primitives["toruses"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["toruses"][i];
			std::string name;
			point3 position{};
			float major_radius = 0.0f;
			float minor_radius = 0.0f;
			std::string materialName;
			uvmapping uv = { 1, 1, 0, 0, 1, 1 };
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("position"))
				position = this->getPoint(primitive["position"]);
			if (primitive.exists("major_radius"))
				primitive.lookupValue("major_radius", major_radius);
			if (primitive.exists("minor_radius"))
				primitive.lookupValue("minor_radius", minor_radius);
			if (primitive.exists("material"))
				primitive.lookupValue("material", materialName);
			if (primitive.exists("uvmapping"))
				uv = this->getUVmapping(primitive["uvmapping"]);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (materialName.empty())
				throw std::runtime_error("Material name is empty");

			if (active)
			{
				builder.addTorus(name.c_str(), position, major_radius, minor_radius, materialName.c_str(), uv, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}

void scene_loader::addVolumePrimitives(const libconfig::Setting& primitives, scene_builder& builder)
{
	if (primitives.exists("volumes"))
	{
		for (int i = 0; i < primitives["volumes"].getLength(); i++)
		{
			const libconfig::Setting& primitive = primitives["volumes"][i];
			std::string name;
			std::string boundary;
			float density = 0.0f;
			color rgb{};
			std::string textureName;
			std::string groupName;
			bool active = true;

			if (primitive.exists("name"))
				primitive.lookupValue("name", name);
			if (primitive.exists("boundary"))
				primitive.lookupValue("boundary", boundary);
			if (primitive.exists("density"))
				primitive.lookupValue("density", density);
			if (primitive.exists("color"))
				rgb = this->getColor(primitive["color"]);
			if (primitive.exists("texture"))
				primitive.lookupValue("texture", textureName);
			if (primitive.exists("group"))
				primitive.lookupValue("group", groupName);
			if (primitive.exists("active"))
				primitive.lookupValue("active", active);

			if (active)
			{
				if (!textureName.empty())
					builder.addVolume(name.c_str(), boundary.c_str(), density, textureName.c_str(), groupName.c_str());
				else
					builder.addVolume(name.c_str(), boundary.c_str(), density, rgb, groupName.c_str());

				applyTransform(primitive, builder, name.c_str());
			}
		}
	}
}



point3 scene_loader::getPoint(const libconfig::Setting& setting)
{
	point3 point;
	point.x = 0.0;
	point.y = 0.0;
	point.z = 0.0;

	if (setting.exists("x"))
	{
		if (!setting.lookupValue("x", point.x))
		{
			int x = 0;
			if (setting.lookupValue("x", x))
			{
				point.x = float(x);
			}
		}
	}

	if (setting.exists("y"))
	{
		if (!setting.lookupValue("y", point.y))
		{
			int y = 0;
			if (setting.lookupValue("y", y))
			{
				point.y = float(y);
			}
		}
	}

	if (setting.exists("z"))
	{
		if (!setting.lookupValue("z", point.z))
		{
			int z = 0;
			if (setting.lookupValue("z", z))
			{
				point.z = float(z);
			}
		}
	}

	return point;
}

vector3 scene_loader::getVector(const libconfig::Setting& setting)
{
	vector3 vector;
	vector.x = 0.0;
	vector.y = 0.0;
	vector.z = 0.0;

	if (setting.exists("x"))
	{
		if (!setting.lookupValue("x", vector.x))
		{
			int x = 0;
			if (setting.lookupValue("x", x))
			{
				vector.x = float(x);
			}
		}
	}

	if (setting.exists("y"))
	{
		if (!setting.lookupValue("y", vector.y))
		{
			int y = 0;
			if (setting.lookupValue("y", y))
			{
				vector.y = float(y);
			}
		}
	}

	if (setting.exists("z"))
	{
		if (!setting.lookupValue("z", vector.z))
		{
			int z = 0;
			if (setting.lookupValue("z", z))
			{
				vector.z = float(z);
			}
		}
	}

	return vector;
}

color scene_loader::getColor(const libconfig::Setting& setting)
{
	int r1 = -1;
	int g1 = -1;
	int b1 = -1;

	// test if format is 0-255
	if (setting.exists("r"))
		setting.lookupValue("r", r1);
	if (setting.exists("g"))
		setting.lookupValue("g", g1);
	if (setting.exists("b"))
		setting.lookupValue("b", b1);



	if (r1 >= 0 && g1 >= 0 && b1 >= 0)
	{
		return color(r1 / 255.0f, g1 / 255.0f, b1 / 255.0f);
	}

	float r2 = 0.0f;
	float g2 = 0.0f;
	float b2 = 0.0f;

	// test if format is 0.0-1.0
	if (setting.exists("r"))
		setting.lookupValue("r", r2);
	if (setting.exists("g"))
		setting.lookupValue("g", g2);
	if (setting.exists("b"))
		setting.lookupValue("b", b2);

	return color(r2, g2, b2);
}

uvmapping scene_loader::getUVmapping(const libconfig::Setting& setting)
{
	uvmapping uv{};

	float scale_u = 1.0f;
	float scale_v = 1.0f;
	float offset_u = 0.0f;
	float offset_v = 0.0f;
	float repeat_u = 1.0f;
	float repeat_v = 1.0f;

	if (setting.exists("scale_u"))
		setting.lookupValue("scale_u", scale_u);
	if (setting.exists("scale_v"))
		setting.lookupValue("scale_v", scale_v);
	if (setting.exists("offset_u"))
		setting.lookupValue("offset_u", offset_u);
	if (setting.exists("offset_v"))
		setting.lookupValue("offset_v", offset_v);
	if (setting.exists("repeat_u"))
		setting.lookupValue("repeat_u", repeat_u);
	if (setting.exists("repeat_v"))
		setting.lookupValue("repeat_v", repeat_v);

	uv.scale_u(scale_u);
	uv.scale_v(scale_v);
	uv.offset_u(offset_u);
	uv.offset_v(offset_v);
	uv.repeat_u(repeat_u);
	uv.repeat_v(repeat_v);

	return uv;
}

rt::transform scene_loader::getTransform(const libconfig::Setting& setting)
{
	rt::transform trs;

	if (setting.exists("translate"))
		trs.setTranslate(this->getVector(setting["translate"]));
	if (setting.exists("rotate"))
		trs.setRotate(this->getVector(setting["rotate"]));
	if (setting.exists("scale"))
		trs.setScale(this->getVector(setting["scale"]));

	return trs;
}