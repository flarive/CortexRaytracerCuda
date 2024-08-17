#pragma once

#include <iostream>
#include <chrono>

class timer
{
public:
    void start();
    void stop();
    double elapsedMilliseconds();
    double elapsedSeconds();
    std::chrono::time_point<std::chrono::system_clock> getStartTime();
    std::chrono::time_point<std::chrono::system_clock> getEndTime();
    void reset();
    std::string format_duration(double dms);
    void displayTime();

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool m_bRunning = false;
};



void timer::start()
{
    m_StartTime = std::chrono::system_clock::now();
    m_bRunning = true;
}

void timer::stop()
{
    m_EndTime = std::chrono::system_clock::now();
    m_bRunning = false;
}

double timer::elapsedMilliseconds()
{
    std::chrono::time_point<std::chrono::system_clock> endTime;

    if (m_bRunning)
    {
        endTime = std::chrono::system_clock::now();
    }
    else
    {
        endTime = m_EndTime;
    }

    return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count());
}

double timer::elapsedSeconds()
{
    return elapsedMilliseconds() / 1000.0;
}

std::chrono::time_point<std::chrono::system_clock> timer::getStartTime()
{
    return m_StartTime;
}

std::chrono::time_point<std::chrono::system_clock> timer::getEndTime()
{
    return m_EndTime;
}

void timer::reset()
{
    m_StartTime = m_EndTime = {};
}

std::string timer::format_duration(double dms)
{
    std::chrono::duration<double, std::milli> ms{ dms };

    auto secs = duration_cast<std::chrono::seconds>(ms);
    auto mins = duration_cast<std::chrono::minutes>(secs);
    secs -= duration_cast<std::chrono::seconds>(mins);
    auto hour = duration_cast<std::chrono::hours>(mins);
    mins -= duration_cast<std::chrono::minutes>(hour);

    std::stringstream ss;
    ss << hour.count() << "h " << mins.count() << "mn " << secs.count() << "s";
    return ss.str();
}

void timer::displayTime()
{
    std::string www = format_duration(elapsedMilliseconds());
    std::cout << "[INFO] Execution Time: " << www.c_str() << std::endl;
}