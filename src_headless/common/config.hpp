#pragma once

// Grid configuration - 256x256 grid
#define GRID_WIDTH 256
#define GRID_HEIGHT 256
#define GRID_SIZE (GRID_WIDTH * GRID_HEIGHT)

// Simulation configuration
#define NUM_FOOD_SOURCES 7
#define FOOD_AMOUNT 2000       // Amount of food at each source
#define NUM_ANTS 5000          // Number of ants in the colony
#define PHEROMONE_DECAY 0.995  // Pheromone decay rate per tick (slower decay)

#define MAX_TICKS 2147483647  // Maximum number of ticks to prevent overflow

// Coordinate system
#define X_MIN (-GRID_WIDTH / 2)
#define Y_MIN (-GRID_HEIGHT / 2)
#define X_MAX (GRID_WIDTH / 2 - 1)
#define Y_MAX (GRID_HEIGHT / 2 - 1)

// Cell states
enum class CellType : int {
    EMPTY = 0,
    NEST = 1,
    FOOD = 2
};

// Ant states
enum class AntState : int {
    SEARCHING = 0,  // Looking for food
    RETURNING = 1   // Returning to nest with food
};

// Direction offsets for 8 neighbors
constexpr int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
