#pragma once

#include <vector>

struct hsv {
    double h;  // angle in degrees
    double s;  // a fraction between 0 and 1
    double v;  // a fraction between 0 and 1
};

struct rgb {
    int r, g, b, a;

    rgb() : r(255), g(255), b(255), a(255) {}
    rgb(int r_in, int g_in, int b_in, int alpha = 255) : r(r_in), g(g_in), b(b_in), a(alpha) {}

    rgb(hsv c, int alpha = 255) : a(alpha) {
        double r_, g_, b_;
        double hh, p, q, t, ff;
        long i;

        if (c.s <= 0.0) {
            r_ = g_ = b_ = c.v;
        } else {
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
                default:
                    r_ = c.v;
                    g_ = p;
                    b_ = q;
                    break;
            }
        }
        r = r_ * 255;
        g = g_ * 255;
        b = b_ * 255;
    }
};
