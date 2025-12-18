#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include "config.hpp"

// Cell structure for the grid
struct Cell {
    CellType type = CellType::EMPTY;
    int food = 0;                 // Amount of food remaining
    double food_pheromone = 0.0;  // Pheromone leading to food
    double nest_pheromone = 0.0;  // Pheromone leading to nest
    int ant_id = -1;              // ID of ant on this cell, -1 if none
};

// Ant structure
struct Ant {
    int x, y;  // Current position
    AntState state = AntState::SEARCHING;
    bool has_food = false;
    double orientation = 0.0;  // Current movement direction (radians)

    Ant() : x(0), y(0), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
    Ant(int x, int y) : x(x), y(y), state(AntState::SEARCHING), has_food(false), orientation(0.0) {}
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
    RNG(unsigned seed = 42) : gen(seed), dist(0.0, 1.0), int_dist(0, GRID_WIDTH - 1) {}

    double random_double() { return dist(gen); }
    double random_double(double min, double max) { return min + (max - min) * dist(gen); }
    int random_int(int min, int max) {
        std::uniform_int_distribution<int> d(min, max);
        return d(gen);
    }
    bool flip_coin() { return dist(gen) < 0.5; }

   private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
    std::uniform_int_distribution<int> int_dist;
};
