#include "controllers/cli.hpp"
#include <cstring>

GameOptions::GameOptions(int args_size, char* args[]) {
    for (size_t i = 0; i < args_size; i++) {
        char* arg = args[i];
        if (!strcmp(arg, "nests") || !strcmp(arg, "--nests") || !strcmp(arg, "-n")) {
            colonies_amount = atoi(args[++i]);
            continue;
        }
        if (!strcmp(arg, "full") || !strcmp(arg, "--full") || !strcmp(arg, "-f")) {
            fullscreen_view = true;
            continue;
        }
        if (!strcmp(arg, "-fn")) {
            fullscreen_view = true;
            colonies_amount = atoi(args[++i]);
            break;
        }
        if (!strcmp(arg, "--width") || !strcmp(arg, "-w")) {
            width = atoi(args[++i]);
            continue;
        }
        if (!strcmp(arg, "--height") || !strcmp(arg, "-h")) {
            height = atoi(args[++i]);
            continue;
        }
        if (!strcmp(arg, "--headless")) {
            headless = true;
            continue;
        }
    }
}