#include "sceneManager.h"

#include "utilities/helpers.h"

#include <iostream>
#include <filesystem>



namespace fs = std::filesystem;

sceneManager::sceneManager()
{
}

void sceneManager::setScenesPath(const std::string path)
{
    m_scenesPath = path;
}


std::vector<scene> sceneManager::listAllScenes()
{
    std::vector<scene> scenes;

    fs::path dir(std::filesystem::current_path());
    fs::path file(this->m_scenesPath);
    fs::path fullexternalProgramPath = dir / file;

    auto fullAbsPath = fs::absolute(fullexternalProgramPath);

    if (fs::exists(fullAbsPath))
    {
        for (const auto& entry : fs::directory_iterator(fullAbsPath))
        {
            scene sc = scene(entry.path().filename().string(), entry.path().generic_string());
            scenes.emplace_back(sc);
        }
    }

    return scenes;
}

std::unique_ptr<sceneSettings> sceneManager::readSceneSettings(std::string filepath)
{
    fs::path dir(std::filesystem::current_path());
    fs::path file(filepath);
    fs::path fullAbsPath = dir / file;

    if (fs::exists(fullAbsPath))
    {
        try
        {
            m_cfg.readFile(fullAbsPath.string());
        }
        catch (const libconfig::ParseException& e)
        {
            std::cerr << "[ERROR] Scene parsing error line " << e.getLine() << " " << e.getError() << std::endl;
            return nullptr;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[ERROR] Scene parsing failed ! " << e.what() << std::endl;
            return nullptr;
        }


        sceneSettings settings;

        const libconfig::Setting& root = m_cfg.getRoot();

        if (root.exists("image"))
        {
            const libconfig::Setting& image = root["image"];
            loadImageConfig(settings, image);
        }

        if (root.exists("camera"))
        {
            const libconfig::Setting& camera = root["camera"];
            loadCameraConfig(settings, camera);
        }

        return std::make_unique<sceneSettings>(settings);

    }

    return nullptr;
}

void sceneManager::loadImageConfig(sceneSettings& settings, const libconfig::Setting& setting)
{
    if (setting.exists("width"))
        settings.width = setting["width"];
    if (setting.exists("height"))
        settings.height = setting["height"];
    if (setting.exists("maxDepth"))
        settings.depth = setting["maxDepth"];
    if (setting.exists("samplesPerPixel"))
        settings.spp = setting["samplesPerPixel"];
}

void sceneManager::loadCameraConfig(sceneSettings& settings, const libconfig::Setting& setting)
{
    if (setting.exists("aspectRatio"))
    {
        std::string zzz = setting["aspectRatio"];
        settings.aspectRatio = zzz;
    }
}
