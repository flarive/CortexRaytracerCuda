#pragma once

#include <iostream>
#include <sstream>

class helpers
{
public:
    static double getRatio(std::string value)
    {
        if (value.empty())
        {
            return 0.0;
        }

        double p1 = 0, p2 = 0;

        std::stringstream test(value);
        std::string segment;

        unsigned int loop = 0;
        while (getline(test, segment, ':'))
        {
            if (loop == 0)
            {
                p1 = stoul(segment, 0, 10);
            }
            else if (loop == 1)
            {
                p2 = stoul(segment, 0, 10);
            }

            loop++;
        }

        if (p1 > 0 && p2 > 0)
        {
            return p1 / p2;
        }

        return 0.0;
    }
};

