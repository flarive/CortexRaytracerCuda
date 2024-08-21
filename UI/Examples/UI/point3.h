#pragma once

class point3
{
public:
    double e[3];

    point3() : e{ 0,0,0 } {}
    point3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }
};
