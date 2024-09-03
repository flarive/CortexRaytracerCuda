#pragma once

#include "output.cuh"

#include "../utilities/interval.cuh"

#define WIN32_LEAN_AND_MEAN // Exclude rarely-used services from Windows headers
#define NOMINMAX // Prevent the definition of min and max macros
#include <windows.h>
#include <iostream>

class namedpipes_output : public output
{
public:
    __host__ int init_output(const size_t dataSize) override;
    __host__ int write_to_output(int x, int y, color pixel_color) const override;
    __host__ int clean_output() override;

private:
    HANDLE m_hPipe = nullptr;
    BOOL m_client_connected = false;
};





__host__ int namedpipes_output::init_output(const size_t dataSize)
{
    const wchar_t* pipeName = L"\\\\.\\pipe\\MyNamedPipe";

    // Create a named pipe
    m_hPipe = CreateNamedPipe(
        pipeName,              // Pipe name
        PIPE_ACCESS_DUPLEX,    // Read/Write access
        PIPE_TYPE_MESSAGE |    // Message type pipe
        PIPE_READMODE_MESSAGE |// Message-read mode
        PIPE_WAIT,             // Blocking mode
        PIPE_UNLIMITED_INSTANCES, // Max instances
        static_cast<DWORD>(dataSize), // Output buffer size
        static_cast<DWORD>(dataSize), // Input buffer size
        0,                     // Client time-out
        nullptr                // Default security attribute
    );

    if (m_hPipe == INVALID_HANDLE_VALUE) {
        std::cerr << "[ERROR] CreateNamedPipe failed. Error: " << GetLastError() << std::endl;
        return 1;
    }

    std::clog << "[INFO] Waiting for client to connect..." << std::endl;

    // Wait for the client to connect
    m_client_connected = ConnectNamedPipe(m_hPipe, nullptr);

    if (!m_client_connected && GetLastError() != ERROR_PIPE_CONNECTED)
    {
        std::cerr << "[ERROR] ConnectNamedPipe failed. Error: " << GetLastError() << std::endl;
        CloseHandle(m_hPipe);
        return 1;
    }

    std::clog << "[INFO] Client connected." << std::endl;

    return 0;
}

__host__ int namedpipes_output::write_to_output(int x, int y, color pixel_color) const
{
    // Write the translated [0,255] value of each color component.
    // Static Variable gets constructed only once no matter how many times the function is called.
    static const interval intensity(0.000, 0.999);


    if (!m_client_connected)
    {
        std::cerr << "[ERROR] No client connected !" << std::endl;
        return 1;
    }



    // Buffer to hold the formatted wide string
    const int bufferSize = 24;
    char message[bufferSize];

    // Format the string using swprintf (forced constant length)
    sprintf(
        message,
        "%05d %05d %03d %03d %03d\0",
        x,
        y,
        static_cast<int>(256 * intensity.clamp(pixel_color.r())),
        static_cast<int>(256 * intensity.clamp(pixel_color.g())),
        static_cast<int>(256 * intensity.clamp(pixel_color.b()))
    );

    //std::cout << "[INFO] Send " << std::string(message).c_str() << std::endl;



    DWORD bytesWritten;

    bool result = WriteFile(
        m_hPipe,                  // Handle to the pipe
        message,                // Buffer to write
        (strlen(message) + 1) * sizeof(char), // Size of buffer
        &bytesWritten,          // Number of bytes written
        nullptr                 // Not overlapped
    );

    if (!result)
    {
        std::cerr << "[ERROR] WriteFile failed. Error: " << GetLastError() << std::endl;
        return 1;
    }
    //else {
    //    std::clog << "[INFO] Sent message to client : " << bytesWritten << "bytes" << std::endl;
    //}

    return 0;
}

__host__ int namedpipes_output::clean_output()
{
    // Close the pipe handle
    CloseHandle(m_hPipe);

    std::cout << "[INFO] Close pipe handle" << std::endl;

    return 0;
}
