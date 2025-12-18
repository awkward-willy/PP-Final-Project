#pragma once

// Grid configuration - 256x256 grid
#define GRID_WIDTH 256
#define GRID_HEIGHT 256
#define GRID_SIZE (GRID_WIDTH * GRID_HEIGHT)

// Simulation configuration
#define NUM_FOOD_SOURCES 10
#define FOOD_AMOUNT 20
#define NUM_ANTS 100
#define PHEROMONE_DECAY 0.99f

// Coordinate system
#define X_MIN (-GRID_WIDTH / 2)
#define Y_MIN (-GRID_HEIGHT / 2)
#define X_MAX (GRID_WIDTH / 2 - 1)
#define Y_MAX (GRID_HEIGHT / 2 - 1)

// Cell states
#define CELL_EMPTY 0
#define CELL_NEST 1
#define CELL_FOOD 2

// Ant states
#define ANT_SEARCHING 0
#define ANT_RETURNING 1

// CUDA block sizes
#define BLOCK_SIZE 256
#define GRID_BLOCK_SIZE 16
