#pragma once

#include <vector>

struct hsv {
    double h;  // angle in degrees
    double s;  // a fraction between 0 and 1
    double v;  // a fraction between 0 and 1
};

struct rgb {
    rgb();
    rgb(double r_in, double g_in, double b_in, double alpha = 1.0);
    rgb(int r_in, int g_in, int b_in, int alpha = 255);
    rgb(hsv c, int alpha = 255);

    int r;
    int g;
    int b;
    int a;
};

double create_base_tint();
rgb get_tint(std::size_t index, std::size_t size, double base_tint, double s, double v);