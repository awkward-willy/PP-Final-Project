// Ant Colony Simulation with OpenMP + SDL2 Visualization
// Uses the same logic as src_headless versions

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Include configuration from config.hpp
#include "config.hpp"

// ============================================================================
// Types - structs and utility functions
// ============================================================================

struct Cell {
    CellType type = CellType::EMPTY;
    int food = 0;
    double food_pheromone = 0.0;
    double nest_pheromone = 0.0;
    int ant_id = -1;
};

struct Ant {
    int x, y;
    AntState state = AntState::SEARCHING;
    bool has_food = false;
    double orientation = 0.0;

    Ant() : x(0), y(0), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
    Ant(int x, int y) : x(x), y(y), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
};

struct SimStats {
    int total_food_collected = 0;
    int total_food_available = 0;
    size_t ticks = 0;
};

// Utility functions
inline int to_index(int x, int y) {
    return (y - Y_MIN) * GRID_WIDTH + (x - X_MIN);
}

inline bool in_bounds(int x, int y) {
    return x >= X_MIN && x <= X_MAX && y >= Y_MIN && y <= Y_MAX;
}

inline double distance(int x1, int y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

inline double distance_squared(int x1, int y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

// Random number generator wrapper
class RNG {
   public:
    RNG(unsigned seed = 42) : gen(seed), dist(0.0, 1.0) {}

    double random_double() { return dist(gen); }
    double random_double(double min, double max) { return min + (max - min) * dist(gen); }
    int random_int(int min, int max) {
        std::uniform_int_distribution<int> d(min, max);
        return d(gen);
    }

   private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
};

// ============================================================================
// Visualization constants
// ============================================================================

constexpr int WINDOW_WIDTH = 2048;
constexpr int WINDOW_HEIGHT = 2048;
constexpr int CELL_SIZE = WINDOW_WIDTH / GRID_WIDTH;

// ============================================================================
// Simulation Class - same logic as src_headless/cpu/main.cpp
// ============================================================================

class AntSimulation {
   public:
    std::vector<Cell> grid;
    std::vector<Ant> ants;
    SimStats stats;
    std::vector<RNG> rngs;  // Per-thread RNGs for OpenMP

    double nest_x, nest_y;
    int nest_left_x, nest_left_y;
    double simulation_time_ms;
    bool simulation_complete;

    AntSimulation(unsigned seed = 42) {
        grid.resize(GRID_SIZE);
        ants.resize(NUM_ANTS);
        simulation_time_ms = 0;
        simulation_complete = false;

        // Create per-thread RNGs
        int num_threads = omp_get_max_threads();
        for (int i = 0; i < num_threads; i++) {
            rngs.emplace_back(seed + i * 1000);
        }

        initialize(seed);
    }

    void initialize(unsigned seed) {
        RNG rng(seed);

        // Clear grid
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = Cell();
        }

        // Create nest at center (2x2 area)
        nest_left_x = 0;
        nest_left_y = 0;
        nest_x = 0.5;
        nest_y = 0.5;

        for (int dx = 0; dx < 2; dx++) {
            for (int dy = 0; dy < 2; dy++) {
                int idx = to_index(nest_left_x + dx, nest_left_y + dy);
                grid[idx].type = CellType::NEST;
                grid[idx].nest_pheromone = 1.0;
            }
        }

        // Calculate nest pheromones for all cells (distance-based)
        double max_distance = std::sqrt((double)(GRID_WIDTH * GRID_WIDTH + GRID_HEIGHT * GRID_HEIGHT));
        for (int y = Y_MIN; y <= Y_MAX; y++) {
            for (int x = X_MIN; x <= X_MAX; x++) {
                int idx = to_index(x, y);
                double dist = distance(x, y, nest_x, nest_y);
                grid[idx].nest_pheromone = 1.0 - dist / max_distance;
            }
        }

        // Place food sources - at radius 30-80 from nest
        std::vector<std::pair<int, int>> food_locations;
        stats.total_food_available = 0;

        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            int best_x = 0, best_y = 0;
            double best_score = -1;

            for (int attempt = 0; attempt < 50; attempt++) {
                double angle = rng.random_double(0, 2 * M_PI);
                double radius = rng.random_double(30, 80);
                int fx = (int)(nest_x + radius * std::cos(angle));
                int fy = (int)(nest_y + radius * std::sin(angle));

                fx = std::max(X_MIN + 5, std::min(X_MAX - 5, fx));
                fy = std::max(Y_MIN + 5, std::min(Y_MAX - 5, fy));

                double score = 0;
                for (auto& loc : food_locations) {
                    score += distance_squared(fx, fy, loc.first, loc.second);
                }
                if (food_locations.empty())
                    score = 1;

                if (score > best_score) {
                    best_score = score;
                    best_x = fx;
                    best_y = fy;
                }
            }

            int idx = to_index(best_x, best_y);
            grid[idx].type = CellType::FOOD;
            grid[idx].food = FOOD_AMOUNT;
            food_locations.push_back({best_x, best_y});
            stats.total_food_available += FOOD_AMOUNT;
        }

        // Spawn ants around nest
        int ant_idx = 0;
        for (int ring = 1; ant_idx < NUM_ANTS && ring < 100; ring++) {
            for (int dx = -ring; dx <= ring && ant_idx < NUM_ANTS; dx++) {
                for (int dy = -ring; dy <= ring && ant_idx < NUM_ANTS; dy++) {
                    if (std::abs(dx) != ring && std::abs(dy) != ring)
                        continue;

                    int ax = nest_left_x + dx;
                    int ay = nest_left_y + dy;

                    if (!in_bounds(ax, ay))
                        continue;

                    int cell_idx = to_index(ax, ay);
                    if (grid[cell_idx].type == CellType::NEST)
                        continue;
                    if (grid[cell_idx].type == CellType::FOOD)
                        continue;
                    if (grid[cell_idx].ant_id != -1)
                        continue;

                    ants[ant_idx].x = ax;
                    ants[ant_idx].y = ay;
                    ants[ant_idx].orientation = rng.random_double(0, 2 * M_PI);
                    grid[cell_idx].ant_id = ant_idx;
                    ant_idx++;
                }
            }
        }

        stats.total_food_collected = 0;
        stats.ticks = 0;
    }

    void decay_pheromones() {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i].food_pheromone *= PHEROMONE_DECAY;
            if (grid[i].food_pheromone < 0.01) {
                grid[i].food_pheromone = 0;
            }
        }
    }

    // ACO algorithm with boundary avoidance
    int find_best_neighbor(Ant& ant, bool searching_for_food, RNG& rng) {
        std::vector<std::pair<int, double>> valid_moves;

        // Check if near boundary - need to turn back
        bool near_edge = (ant.x <= X_MIN + 3 || ant.x >= X_MAX - 3 ||
                          ant.y <= Y_MIN + 3 || ant.y >= Y_MAX - 3);

        for (int d = 0; d < 8; d++) {
            int nx = ant.x + DX[d];
            int ny = ant.y + DY[d];

            if (!in_bounds(nx, ny))
                continue;

            int idx = to_index(nx, ny);

            // Allow multiple ants per cell - no collision check
            // This prevents traffic jams on pheromone trails

            double score = 1.0;  // Base score

            // Boundary avoidance - strong preference to move toward center when near edge
            if (near_edge) {
                // Calculate direction toward center (nest at 0,0)
                double to_center_x = nest_x - ant.x;
                double to_center_y = nest_y - ant.y;
                double dot = DX[d] * to_center_x + DY[d] * to_center_y;
                if (dot > 0) {
                    score += 5.0;  // Strong reward for moving toward center
                } else {
                    score *= 0.1;  // Strong penalty for moving toward edge
                }
            }

            if (searching_for_food) {
                // Immediately go to food if found
                if (grid[idx].type == CellType::FOOD && grid[idx].food > 0) {
                    return d;
                }

                // ACO: Follow food pheromone trails (stronger weight)
                double pheromone = grid[idx].food_pheromone;
                score += std::pow(pheromone + 0.1, 2.0) * 8.0;

                // Momentum: prefer continuing in current direction
                double angle = std::atan2(ny - ant.y, nx - ant.x);
                double angle_diff = std::abs(angle - ant.orientation);
                if (angle_diff > M_PI)
                    angle_diff = 2 * M_PI - angle_diff;
                double momentum = (M_PI - angle_diff) / M_PI;
                score += momentum * 2.0;

                // Random exploration component
                score += rng.random_double() * 3.0;
            } else {
                // Returning to nest with food - go directly to nest!
                if (grid[idx].type == CellType::NEST) {
                    return d;
                }

                // Strong heuristic: prefer moving towards nest (this is the main driver)
                double current_dist = distance(ant.x, ant.y, nest_x, nest_y);
                double new_dist = distance(nx, ny, nest_x, nest_y);

                // Strongly reward moving closer to nest
                if (new_dist < current_dist) {
                    score += 50.0;  // Strong reward for getting closer
                } else {
                    score *= 0.1;  // Strong penalty for moving away
                }

                // Additional inverse distance heuristic
                if (new_dist > 0) {
                    score += (1.0 / new_dist) * 30.0;
                }

                // Very small random component (almost deterministic return)
                score += rng.random_double() * 0.2;
            }

            valid_moves.push_back({d, score});
        }

        if (valid_moves.empty())
            return -1;

        // Roulette wheel selection
        double total_score = 0;
        for (auto& move : valid_moves) {
            total_score += move.second;
        }

        double r = rng.random_double() * total_score;
        double cumulative = 0;
        for (auto& move : valid_moves) {
            cumulative += move.second;
            if (r <= cumulative) {
                return move.first;
            }
        }

        return valid_moves.back().first;
    }

    void update_ant(int ant_id, RNG& rng) {
        Ant& ant = ants[ant_id];

        bool searching = (ant.state == AntState::SEARCHING);
        int best_dir = find_best_neighbor(ant, searching, rng);

        if (best_dir == -1) {
            ant.orientation = rng.random_double(0, 2 * M_PI);
            return;
        }

        int nx = ant.x + DX[best_dir];
        int ny = ant.y + DY[best_dir];
        int new_idx = to_index(nx, ny);
        int old_idx = to_index(ant.x, ant.y);

        if (searching && grid[new_idx].type == CellType::FOOD && grid[new_idx].food > 0) {
#pragma omp atomic
            grid[new_idx].food--;
            ant.has_food = true;
            ant.state = AntState::RETURNING;
            ant.orientation = std::atan2(nest_y - ny, nest_x - nx);
        }

        if (!searching && grid[new_idx].type == CellType::NEST) {
            if (ant.has_food) {
#pragma omp atomic
                stats.total_food_collected++;
                ant.has_food = false;
                grid[old_idx].food_pheromone += 5.0;  // Stronger pheromone deposit
            }
            ant.state = AntState::SEARCHING;
            ant.orientation = rng.random_double(0, 2 * M_PI);
        }

        if (ant.state == AntState::RETURNING && ant.has_food) {
            grid[old_idx].food_pheromone += 3.0;  // Stronger trail
        }

        // Move ant (no ant_id tracking needed - multiple ants can share cells)
        ant.x = nx;
        ant.y = ny;

        if (ant.state == AntState::SEARCHING) {
            ant.orientation = std::atan2(DY[best_dir], DX[best_dir]);
        }
    }

    void relocate_food() {
        // Clear existing food
        for (int i = 0; i < GRID_SIZE; i++) {
            if (grid[i].type == CellType::FOOD) {
                grid[i].type = CellType::EMPTY;
                grid[i].food = 0;
            }
        }

        // Place new food sources
        std::vector<std::pair<int, int>> food_locations;
        RNG& rng = rngs[0];

        // Calculate dynamic radius based on grid size
        // Use 30% to 80% of the half-width (distance from center to edge)
        double max_radius = std::min(GRID_WIDTH, GRID_HEIGHT) / 2.0 * 0.8;
        double min_radius = std::min(GRID_WIDTH, GRID_HEIGHT) / 2.0 * 0.3;

        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            int best_x = 0, best_y = 0;
            double best_score = -1;

            for (int attempt = 0; attempt < 50; attempt++) {
                double angle = rng.random_double(0, 2 * M_PI);
                double radius = rng.random_double(min_radius, max_radius);
                int fx = (int)(nest_x + radius * std::cos(angle));
                int fy = (int)(nest_y + radius * std::sin(angle));

                fx = std::max(X_MIN + 2, std::min(X_MAX - 2, fx));
                fy = std::max(Y_MIN + 2, std::min(Y_MAX - 2, fy));

                double score = 0;
                for (auto& loc : food_locations) {
                    score += distance_squared(fx, fy, loc.first, loc.second);
                }
                if (food_locations.empty())
                    score = 1;

                if (score > best_score) {
                    best_score = score;
                    best_x = fx;
                    best_y = fy;
                }
            }

            int idx = to_index(best_x, best_y);
            grid[idx].type = CellType::FOOD;
            grid[idx].food = FOOD_AMOUNT;
            food_locations.push_back({best_x, best_y});
            stats.total_food_available += FOOD_AMOUNT;
        }

        // Ensure simulation continues
        simulation_complete = false;
    }

    void tick() {
        auto start = std::chrono::high_resolution_clock::now();

        stats.ticks++;

        // Relocate food every 100,000 ticks
        if (stats.ticks % 100000 == 0) {
            relocate_food();
        }

        decay_pheromones();

        // Sequential update (same as CPU version for correctness)
        // OpenMP parallelization would require careful handling of ant_id conflicts
        for (int i = 0; i < NUM_ANTS; i++) {
            update_ant(i, rngs[0]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        simulation_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

        if (stats.total_food_collected >= stats.total_food_available) {
            simulation_complete = true;
            std::cout << "\n=== Simulation Complete ===" << std::endl;
            std::cout << "Collected " << stats.total_food_collected << "/"
                      << stats.total_food_available << " food in " << stats.ticks << " ticks" << std::endl;
            std::cout << "Total simulation time: " << simulation_time_ms << " ms" << std::endl;
            std::cout << "Average time per tick: " << (simulation_time_ms / stats.ticks) << " ms" << std::endl;
        }
    }

    bool is_complete() const {
        return simulation_complete;
    }
};

// ============================================================================
// View Class - SDL2 Visualization
// ============================================================================

class View {
   public:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    bool running;
    bool paused;
    int simulation_speed;

    AntSimulation sim;

    View() : window(nullptr), renderer(nullptr), font(nullptr), running(true), paused(false), simulation_speed(1) {}

    ~View() {
        cleanup();
    }

    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
            return false;
        }

        if (TTF_Init() < 0) {
            std::cerr << "TTF init failed: " << TTF_GetError() << std::endl;
            return false;
        }

        window = SDL_CreateWindow(
            "Ant Colony Simulation (Visualization)",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH, WINDOW_HEIGHT,
            SDL_WINDOW_SHOWN);

        if (!window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer) {
            renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        }
        if (!renderer) {
            renderer = SDL_CreateRenderer(window, -1, 0);
        }
        if (!renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        const char* font_paths[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            nullptr};

        for (int i = 0; font_paths[i] != nullptr; ++i) {
            font = TTF_OpenFont(font_paths[i], 14);
            if (font)
                break;
        }

        if (!font) {
            std::cerr << "Warning: Could not load font, HUD will be disabled" << std::endl;
        }

        return true;
    }

    void cleanup() {
        if (font)
            TTF_CloseFont(font);
        if (renderer)
            SDL_DestroyRenderer(renderer);
        if (window)
            SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
    }

    void handleEvents() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                    case SDLK_q:
                        running = false;
                        break;
                    case SDLK_SPACE:
                        paused = !paused;
                        break;
                    case SDLK_r:
                        sim = AntSimulation();
                        break;
                    case SDLK_UP:
                        simulation_speed = std::min(simulation_speed * 2, 64);
                        break;
                    case SDLK_DOWN:
                        simulation_speed = std::max(simulation_speed / 2, 1);
                        break;
                }
            }
        }
    }

    // Convert simulation coordinates to screen coordinates
    int screenX(int x) {
        return (x - X_MIN) * CELL_SIZE;
    }

    int screenY(int y) {
        return (y - Y_MIN) * CELL_SIZE;
    }

    void render() {
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
        SDL_RenderClear(renderer);

        // Render grid cells
        for (int y = Y_MIN; y <= Y_MAX; y++) {
            for (int x = X_MIN; x <= X_MAX; x++) {
                int idx = to_index(x, y);
                const Cell& cell = sim.grid[idx];

                SDL_Rect rect = {screenX(x), screenY(y), CELL_SIZE, CELL_SIZE};

                if (cell.type == CellType::NEST) {
                    // Nest - brown
                    SDL_SetRenderDrawColor(renderer, 139, 69, 19, 255);
                    SDL_RenderFillRect(renderer, &rect);
                } else if (cell.type == CellType::FOOD && cell.food > 0) {
                    // Food - green (intensity based on amount)
                    int intensity = std::min(255, 100 + cell.food * 7);
                    SDL_SetRenderDrawColor(renderer, 0, intensity, 0, 255);
                    SDL_RenderFillRect(renderer, &rect);
                } else if (cell.food_pheromone > 0.01) {
                    // Food pheromone trail - red
                    int intensity = std::min(255, (int)(cell.food_pheromone * 50));
                    SDL_SetRenderDrawColor(renderer, intensity, 0, 0, 255);
                    SDL_RenderFillRect(renderer, &rect);
                }
            }
        }

        // Render ants
        for (const Ant& ant : sim.ants) {
            SDL_Rect rect = {screenX(ant.x), screenY(ant.y), CELL_SIZE, CELL_SIZE};

            if (ant.has_food) {
                // Carrying food - yellow
                SDL_SetRenderDrawColor(renderer, 255, 200, 0, 255);
            } else {
                // Searching - white
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            }

            SDL_RenderFillRect(renderer, &rect);
        }

        // Render HUD
        renderHUD();

        SDL_RenderPresent(renderer);
    }

    void renderHUD() {
        if (!font)
            return;

        SDL_Color white = {255, 255, 255, 255};
        SDL_Color yellow = {255, 255, 0, 255};

        char text[256];
        snprintf(text, sizeof(text),
                 "Tick: %zu | Food: %d/%d | Speed: %dx | %s",
                 sim.stats.ticks, sim.stats.total_food_collected,
                 sim.stats.total_food_available,
                 simulation_speed, paused ? "PAUSED" : "RUNNING");

        renderText(text, 10, 10, sim.simulation_complete ? yellow : white);

        snprintf(text, sizeof(text),
                 "Controls: SPACE=Pause, R=Reset, UP/DOWN=Speed, Q=Quit");
        renderText(text, 10, 30, white);

        if (sim.stats.ticks > 0) {
            double avg_ms = sim.simulation_time_ms / sim.stats.ticks;
            snprintf(text, sizeof(text),
                     "Avg tick: %.3f ms | Threads: %d",
                     avg_ms, omp_get_max_threads());
            renderText(text, 10, 50, white);
        }

        if (sim.simulation_complete) {
            snprintf(text, sizeof(text),
                     "COMPLETE! Total time: %.1f ms", sim.simulation_time_ms);
            renderText(text, 10, 70, yellow);
        }
    }

    void renderText(const char* text, int x, int y, SDL_Color color) {
        SDL_Surface* surface = TTF_RenderText_Blended(font, text, color);
        if (surface) {
            SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
            if (texture) {
                SDL_Rect dst = {x, y, surface->w, surface->h};
                SDL_RenderCopy(renderer, texture, nullptr, &dst);
                SDL_DestroyTexture(texture);
            }
            SDL_FreeSurface(surface);
        }
    }

    void run() {
        const int FPS = 60;
        const int FRAME_DELAY = 1000 / FPS;

        std::cout << "=== Ant Colony Simulation (Visualization) ===" << std::endl;
        std::cout << "Grid: " << GRID_WIDTH << "x" << GRID_HEIGHT << std::endl;
        std::cout << "Ants: " << NUM_ANTS << std::endl;
        std::cout << "Food sources: " << NUM_FOOD_SOURCES << std::endl;
        std::cout << "Food per source: " << FOOD_AMOUNT << std::endl;
        std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;
        std::cout << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  SPACE - Pause/Resume" << std::endl;
        std::cout << "  R     - Reset simulation" << std::endl;
        std::cout << "  UP    - Increase speed" << std::endl;
        std::cout << "  DOWN  - Decrease speed" << std::endl;
        std::cout << "  Q/ESC - Quit" << std::endl;
        std::cout << "==============================================" << std::endl;

        while (running) {
            Uint32 frame_start = SDL_GetTicks();

            handleEvents();

            if (!paused && !sim.is_complete()) {
                for (int i = 0; i < simulation_speed; ++i) {
                    sim.tick();
                    if (sim.is_complete())
                        break;
                }
            }

            render();

            Uint32 frame_time = SDL_GetTicks() - frame_start;
            if (frame_time < static_cast<Uint32>(FRAME_DELAY)) {
                SDL_Delay(FRAME_DELAY - frame_time);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads();
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
        omp_set_num_threads(num_threads);
    }

    std::cout << "Starting with " << num_threads << " OpenMP threads" << std::endl;

    View view;
    if (!view.init()) {
        std::cerr << "Failed to initialize view" << std::endl;
        return 1;
    }

    view.run();

    return 0;
}
