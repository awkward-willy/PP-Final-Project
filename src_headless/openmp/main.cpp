#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include "../common/config.hpp"
#include "../common/types.hpp"

class AntSimulationOpenMP {
   private:
    std::vector<Cell> grid;
    std::vector<Ant> ants;
    std::vector<int> ant_new_positions;  // New position for each ant (direction index)
    SimStats stats;
    std::atomic<int> food_collected{0};

    // Thread-local RNGs
    std::vector<RNG> thread_rngs;

    // Nest center coordinates
    double nest_x, nest_y;
    int nest_left_x, nest_left_y;

   public:
    AntSimulationOpenMP(unsigned seed = 42) {
        grid.resize(GRID_SIZE);
        ants.resize(NUM_ANTS);
        ant_new_positions.resize(NUM_ANTS);

        // Initialize thread-local RNGs
        int num_threads = omp_get_max_threads();
        for (int i = 0; i < num_threads; i++) {
            thread_rngs.emplace_back(seed + i);
        }

        initialize(seed);
    }

    void initialize(unsigned seed) {
        RNG main_rng(seed);

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

        // Calculate nest pheromones for all cells (distance-based) - parallelized
        double max_distance = std::sqrt((double)(GRID_WIDTH * GRID_WIDTH + GRID_HEIGHT * GRID_HEIGHT));

#pragma omp parallel for collapse(2)
        for (int y = Y_MIN; y <= Y_MAX; y++) {
            for (int x = X_MIN; x <= X_MAX; x++) {
                int idx = to_index(x, y);
                double dist = distance(x, y, nest_x, nest_y);
                grid[idx].nest_pheromone = 1.0 - dist / max_distance;
            }
        }

        // Place food sources - closer to nest for faster completion
        std::vector<std::pair<int, int>> food_locations;
        stats.total_food_available = 0;

        // Calculate dynamic radius based on grid size (30% to 80% of half-width)
        double max_radius = std::min(GRID_WIDTH, GRID_HEIGHT) / 2.0 * 0.8;
        double min_radius = std::min(GRID_WIDTH, GRID_HEIGHT) / 2.0 * 0.3;

        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            int best_x = 0, best_y = 0;
            double best_score = -1;

            for (int attempt = 0; attempt < 50; attempt++) {
                // Place food at dynamic radius from nest
                double angle = main_rng.random_double(0, 2 * M_PI);
                double radius = main_rng.random_double(min_radius, max_radius);
                int fx = (int)(nest_x + radius * std::cos(angle));
                int fy = (int)(nest_y + radius * std::sin(angle));

                // Clamp to bounds
                fx = std::max(X_MIN + 2, std::min(X_MAX - 2, fx));
                fy = std::max(Y_MIN + 2, std::min(Y_MAX - 2, fy));

                double score = 0;

                // Add distance from other food sources (prefer spacing)
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
        for (int ring = 1; ant_idx < NUM_ANTS && ring < 20; ring++) {
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
                    ants[ant_idx].orientation = main_rng.random_double(0, 2 * M_PI);
                    grid[cell_idx].ant_id = ant_idx;
                    ant_idx++;
                }
            }
        }

        stats.total_food_collected = 0;
        stats.ticks = 0;
        food_collected = 0;
    }

    void decay_pheromones() {
#pragma omp parallel for
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i].food_pheromone *= PHEROMONE_DECAY;
            if (grid[i].food_pheromone < 0.01) {
                grid[i].food_pheromone = 0;
            }
        }
    }

    // ACO algorithm with boundary avoidance
    int find_best_neighbor(int ant_id, RNG& rng) {
        Ant& ant = ants[ant_id];
        std::vector<std::pair<int, double>> valid_moves;

        bool searching = (ant.state == AntState::SEARCHING);

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

            double score = 1.0;  // Base score

            // Boundary avoidance
            if (near_edge) {
                double to_center_x = nest_x - ant.x;
                double to_center_y = nest_y - ant.y;
                double dot = DX[d] * to_center_x + DY[d] * to_center_y;
                if (dot > 0) {
                    score += 5.0;
                } else {
                    score *= 0.1;
                }
            }

            if (searching) {
                // Immediately go to food if found
                if (grid[idx].type == CellType::FOOD && grid[idx].food > 0) {
                    return d;
                }

                // ACO: Follow food pheromone trails
                double pheromone = grid[idx].food_pheromone;
                score += std::pow(pheromone + 0.1, 2.0) * 8.0;

                // Momentum
                double angle = std::atan2(ny - ant.y, nx - ant.x);
                double angle_diff = std::abs(angle - ant.orientation);
                if (angle_diff > M_PI)
                    angle_diff = 2 * M_PI - angle_diff;
                double momentum = (M_PI - angle_diff) / M_PI;
                score += momentum * 2.0;

                // Random exploration
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

    void tick() {
        stats.ticks++;

        // Decay pheromones (parallel)
        decay_pheromones();

        // Phase 1: Calculate new positions for all ants (parallel)
        // Each ant decides where to move based on current grid state
#pragma omp parallel for schedule(static)
        for (int i = 0; i < NUM_ANTS; i++) {
            int thread_id = omp_get_thread_num();
            ant_new_positions[i] = find_best_neighbor(i, thread_rngs[thread_id]);
        }

        // Phase 2: Apply moves (sequential to maintain correctness)
        for (int i = 0; i < NUM_ANTS; i++) {
            Ant& ant = ants[i];
            int best_dir = ant_new_positions[i];

            if (best_dir == -1) {
                ant.orientation = thread_rngs[0].random_double(0, 2 * M_PI);
                continue;
            }

            int nx = ant.x + DX[best_dir];
            int ny = ant.y + DY[best_dir];
            int new_idx = to_index(nx, ny);
            int old_idx = to_index(ant.x, ant.y);

            // Allow multiple ants per cell - no collision check needed
            bool searching = (ant.state == AntState::SEARCHING);

            // Check if we found food while searching
            if (searching && grid[new_idx].type == CellType::FOOD && grid[new_idx].food > 0) {
                grid[new_idx].food--;
                ant.has_food = true;
                ant.state = AntState::RETURNING;
                ant.orientation = std::atan2(nest_y - ny, nest_x - nx);
            }

            // Check if we reached nest while returning
            if (!searching && grid[new_idx].type == CellType::NEST) {
                if (ant.has_food) {
                    food_collected++;
                    ant.has_food = false;
                    grid[old_idx].food_pheromone += 5.0;
                }
                ant.state = AntState::SEARCHING;
                ant.orientation = thread_rngs[0].random_double(0, 2 * M_PI);
            }

            // If returning with food, leave pheromone trail
            if (ant.state == AntState::RETURNING && ant.has_food) {
                grid[old_idx].food_pheromone += 3.0;
            }

            // Move ant (allow multiple ants per cell)
            ant.x = nx;
            ant.y = ny;

            // Update orientation
            if (ant.state == AntState::SEARCHING) {
                ant.orientation = std::atan2(DY[best_dir], DX[best_dir]);
            }
        }

        stats.total_food_collected = food_collected.load();
    }

    bool is_complete() const {
        return stats.total_food_collected >= stats.total_food_available;
    }

    const SimStats& get_stats() const {
        return stats;
    }

    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();

        while (!is_complete() && stats.ticks < MAX_TICKS) {
            tick();

            if (stats.ticks % 1000 == 0) {
                std::cout << "Tick " << stats.ticks
                          << ": Collected " << stats.total_food_collected
                          << "/" << stats.total_food_available << " food" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\n=== Simulation Complete (OpenMP Version) ===" << std::endl;
        std::cout << "Total ticks: " << stats.ticks << std::endl;
        std::cout << "Food collected: " << stats.total_food_collected
                  << "/" << stats.total_food_available << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Time per tick: " << (double)duration.count() / stats.ticks << " ms" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    unsigned seed = 42;
    int num_threads = omp_get_max_threads();

    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_threads = std::atoi(argv[2]);
        omp_set_num_threads(num_threads);
    }

    std::cout << "=== Ant Colony Simulation (OpenMP Version) ===" << std::endl;
    std::cout << "Grid size: " << GRID_WIDTH << " x " << GRID_HEIGHT << std::endl;
    std::cout << "Number of ants: " << NUM_ANTS << std::endl;
    std::cout << "Number of food sources: " << NUM_FOOD_SOURCES << std::endl;
    std::cout << "Food per source: " << FOOD_AMOUNT << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "=============================================\n"
              << std::endl;

    AntSimulationOpenMP simulation(seed);
    simulation.run();

    return 0;
}
