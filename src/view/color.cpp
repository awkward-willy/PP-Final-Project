#include "view/color.hpp"
#include "controllers/random.hpp"

double create_base_tint() {
    return random_double(0, 360);
}

rgb get_tint(std::size_t index, std::size_t size, double base_tint, double s, double v) {
    double tint = ((int)(base_tint + (360.0 / size) * index) % 360);
    return (hsv{tint, s, v});
}

rgb::rgb() : r(255), g(255), b(255), a(255) {}

rgb::rgb(double r_in, double g_in, double b_in, double alpha) : r(r_in * 255), g(g_in * 255), b(b_in * 255), a(alpha * 255) {}

rgb::rgb(int r_in, int g_in, int b_in, int alpha) : r(r_in), g(g_in), b(b_in), a(alpha) {}

// https://stackoverflow.com/a/6930407
rgb::rgb(hsv c, int alpha) : a(alpha) {
    double r_;
    double g_;
    double b_;
    double hh, p, q, t, ff;
    long i;
    if (c.s <= 0.0) {  // < is bogus, just shuts up warnings
        r_ = c.v;
        g_ = c.v;
        b_ = c.v;
    }
    hh = c.h;
    if (hh >= 360.0)
        hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = c.v * (1.0 - c.s);
    q = c.v * (1.0 - (c.s * ff));
    t = c.v * (1.0 - (c.s * (1.0 - ff)));

    switch (i) {
        case 0:
            r_ = c.v;
            g_ = t;
            b_ = p;
            break;
        case 1:
            r_ = q;
            g_ = c.v;
            b_ = p;
            break;
        case 2:
            r_ = p;
            g_ = c.v;
            b_ = t;
            break;

        case 3:
            r_ = p;
            g_ = q;
            b_ = c.v;
            break;
        case 4:
            r_ = t;
            g_ = p;
            b_ = c.v;
            break;
        case 5:
        default:
            r_ = c.v;
            g_ = p;
            b_ = q;
            break;
    }
    r = r_ * 255;
    g = g_ * 255;
    b = b_ * 255;
}