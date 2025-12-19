#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "config.hpp"

// Cell types
enum class CellType : int {
    EMPTY = 0,
    NEST = 1,
    FOOD = 2
};

// Ant states
enum class AntState : int {
    SEARCHING = 0,
    RETURNING = 1
};

// Cell structure for the grid
struct Cell {
    CellType type = CellType::EMPTY;
    int food = 0;
    double food_pheromone = 0.0;
    double nest_pheromone = 0.0;
    int ant_id = -1;

    bool is_void() const { return type == CellType::EMPTY && ant_id == -1; }
    bool is_nest() const { return type == CellType::NEST; }
    bool has_sugar() const { return type == CellType::FOOD && food > 0; }
    bool has_ant() const { return ant_id != -1; }
};

// Ant structure
struct Ant {
    int x, y;
    AntState state = AntState::SEARCHING;
    bool has_food = false;
    double orientation = 0.0;

    Ant() : x(0), y(0), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
    Ant(int x, int y) : x(x), y(y), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
};

// Coordinates structure for compatibility
struct Coordinates {
    int x, y;
    Coordinates() : x(0), y(0) {}
    Coordinates(int x, int y) : x(x), y(y) {}
};

// Simulation statistics
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
