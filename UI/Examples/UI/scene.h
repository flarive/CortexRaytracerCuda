#pragma once

#include <string>

struct scene
{
public:

    scene(const std::string _name, std::string _path) : m_name(_name), m_path(_path)
    {
    }

    std::string getName()
    {
        return m_name;
    }

    std::string getPath()
    {
        return m_path;
    }

private:
    std::string m_name;
    std::string m_path;
};
