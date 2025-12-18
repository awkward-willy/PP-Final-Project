#include "config_cuda.cuh"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Direction offsets for 8 neighbors
__constant__ int d_DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__constant__ int d_DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

// Device structures
struct CellGPU {
    int type;           // CELL_EMPTY, CELL_NEST, CELL_FOOD
    int food;
    float food_pheromone;
    float nest_pheromone;
    int ant_id;
};

struct AntGPU {
    int x, y;
    int state;          // ANT_SEARCHING, ANT_RETURNING
    int has_food;
    float orientation;
    int new_dir;        // Direction to move (-1 if no move)
};

// Utility device functions
__device__ __forceinline__ int to_index_gpu(int x, int y) {
    return (y - Y_MIN) * GRID_WIDTH + (x - X_MIN);
}

__device__ __forceinline__ bool in_bounds_gpu(int x, int y) {
    return x >= X_MIN && x <= X_MAX && y >= Y_MIN && y <= Y_MAX;
}

__device__ __forceinline__ float distance_sq_gpu(int x1, int y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

// Kernel to decay pheromones
__global__ void decay_pheromones_kernel(CellGPU* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GRID_SIZE) return;
    
    grid[idx].food_pheromone *= PHEROMONE_DECAY;
    if (grid[idx].food_pheromone < 0.01f) {
        grid[idx].food_pheromone = 0.0f;
    }
}

// Kernel to find best move for each ant (parallel, read-only grid access)
__global__ void find_moves_kernel(CellGPU* grid, AntGPU* ants, curandState* rand_states, 
                                  float nest_x, float nest_y) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= NUM_ANTS) return;
    
    AntGPU& ant = ants[ant_id];
    curandState localState = rand_states[ant_id];
    
    bool searching = (ant.state == ANT_SEARCHING);
    float current_dist_sq = distance_sq_gpu(ant.x, ant.y, nest_x, nest_y);
    
    // Store valid moves and scores
    int valid_dirs[8];
    float valid_scores[8];
    int num_valid = 0;
    float total_score = 0.0f;
    
    for (int d = 0; d < 8; d++) {
        int nx = ant.x + d_DX[d];
        int ny = ant.y + d_DY[d];
        
        if (!in_bounds_gpu(nx, ny)) continue;
        
        int idx = to_index_gpu(nx, ny);
        
        // Can't move to cell with another ant
        if (grid[idx].ant_id != -1) continue;
        
        float score = 1.0f;
        
        if (searching) {
            // Immediate return for food
            if (grid[idx].type == CELL_FOOD && grid[idx].food > 0) {
                ant.new_dir = d;
                rand_states[ant_id] = localState;
                return;
            }
            
            score += grid[idx].food_pheromone * 5.0f;
            
            float new_dist_sq = distance_sq_gpu(nx, ny, nest_x, nest_y);
            if (new_dist_sq > current_dist_sq) {
                score += 2.0f;
            }
            
            float angle = atan2f((float)(ny - ant.y), (float)(nx - ant.x));
            float angle_diff = fabsf(angle - ant.orientation);
            if (angle_diff > M_PI) angle_diff = 2 * M_PI - angle_diff;
            score += (M_PI - angle_diff) / M_PI * 3.0f;
            
            score += curand_uniform(&localState) * 2.0f;
        } else {
            // Immediate return for nest
            if (grid[idx].type == CELL_NEST) {
                ant.new_dir = d;
                rand_states[ant_id] = localState;
                return;
            }
            
            score = grid[idx].nest_pheromone * 10.0f;
            score += curand_uniform(&localState) * 0.5f;
        }
        
        valid_dirs[num_valid] = d;
        valid_scores[num_valid] = score;
        total_score += score;
        num_valid++;
    }
    
    if (num_valid == 0) {
        ant.new_dir = -1;
        rand_states[ant_id] = localState;
        return;
    }
    
    // Probabilistic selection
    float r = curand_uniform(&localState) * total_score;
    float cumulative = 0.0f;
    int selected = valid_dirs[num_valid - 1];
    
    for (int i = 0; i < num_valid; i++) {
        cumulative += valid_scores[i];
        if (r <= cumulative) {
            selected = valid_dirs[i];
            break;
        }
    }
    
    ant.new_dir = selected;
    rand_states[ant_id] = localState;
}

// Kernel to apply moves (run with 1 thread for sequential consistency)
// This is a bottleneck but necessary for correctness with ant collisions
__global__ void apply_moves_kernel(CellGPU* grid, AntGPU* ants, curandState* rand_states,
                                   int* food_collected, float nest_x, float nest_y) {
    // Single thread execution
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    for (int i = 0; i < NUM_ANTS; i++) {
        AntGPU& ant = ants[i];
        int best_dir = ant.new_dir;
        
        if (best_dir == -1) {
            ant.orientation = curand_uniform(&rand_states[i]) * 2 * M_PI;
            continue;
        }
        
        int nx = ant.x + d_DX[best_dir];
        int ny = ant.y + d_DY[best_dir];
        int new_idx = to_index_gpu(nx, ny);
        int old_idx = to_index_gpu(ant.x, ant.y);
        
        // Check if cell is still free
        if (grid[new_idx].ant_id != -1) continue;
        
        bool searching = (ant.state == ANT_SEARCHING);
        
        // Check if we found food
        if (searching && grid[new_idx].type == CELL_FOOD && grid[new_idx].food > 0) {
            grid[new_idx].food--;
            ant.has_food = 1;
            ant.state = ANT_RETURNING;
            // Set orientation towards nest
            ant.orientation = atan2f(nest_y - ny, nest_x - nx);
        }
        
        // Check if we reached nest
        if (!searching && grid[new_idx].type == CELL_NEST) {
            if (ant.has_food) {
                atomicAdd(food_collected, 1);
                ant.has_food = 0;
                grid[old_idx].food_pheromone += 1.0f;
            }
            ant.state = ANT_SEARCHING;
            ant.orientation = curand_uniform(&rand_states[i]) * 2 * M_PI;
        }
        
        // Leave pheromone trail if returning with food
        if (ant.state == ANT_RETURNING && ant.has_food) {
            grid[old_idx].food_pheromone += 1.0f;
        }
        
        // Move ant
        grid[old_idx].ant_id = -1;
        grid[new_idx].ant_id = i;
        ant.x = nx;
        ant.y = ny;
        
        // Update orientation based on movement (only for searching ants)
        if (ant.state == ANT_SEARCHING) {
            ant.orientation = atan2f((float)d_DY[best_dir], (float)d_DX[best_dir]);
        }
    }
}

// Initialize curand states
__global__ void init_curand_kernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_ANTS) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Host class for simulation
class AntSimulationCUDA {
private:
    std::vector<CellGPU> h_grid;
    std::vector<AntGPU> h_ants;
    
    CellGPU* d_grid;
    AntGPU* d_ants;
    curandState* d_rand_states;
    int* d_food_collected;
    
    float nest_x, nest_y;
    int nest_left_x, nest_left_y;
    
    int total_food_available;
    int total_food_collected;
    size_t ticks;
    
public:
    AntSimulationCUDA(unsigned seed = 42) : total_food_collected(0), ticks(0) {
        h_grid.resize(GRID_SIZE);
        h_ants.resize(NUM_ANTS);
        
        // Allocate device memory
        cudaMalloc(&d_grid, GRID_SIZE * sizeof(CellGPU));
        cudaMalloc(&d_ants, NUM_ANTS * sizeof(AntGPU));
        cudaMalloc(&d_rand_states, NUM_ANTS * sizeof(curandState));
        cudaMalloc(&d_food_collected, sizeof(int));
        
        initialize(seed);
    }
    
    ~AntSimulationCUDA() {
        cudaFree(d_grid);
        cudaFree(d_ants);
        cudaFree(d_rand_states);
        cudaFree(d_food_collected);
    }
    
    void initialize(unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Clear grid
        for (int i = 0; i < GRID_SIZE; i++) {
            h_grid[i].type = CELL_EMPTY;
            h_grid[i].food = 0;
            h_grid[i].food_pheromone = 0.0f;
            h_grid[i].nest_pheromone = 0.0f;
            h_grid[i].ant_id = -1;
        }
        
        // Create nest at center
        nest_left_x = 0;
        nest_left_y = 0;
        nest_x = 0.5f;
        nest_y = 0.5f;
        
        for (int dx = 0; dx < 2; dx++) {
            for (int dy = 0; dy < 2; dy++) {
                int idx = (nest_left_y + dy - Y_MIN) * GRID_WIDTH + (nest_left_x + dx - X_MIN);
                h_grid[idx].type = CELL_NEST;
                h_grid[idx].nest_pheromone = 1.0f;
            }
        }
        
        // Calculate nest pheromones
        float max_distance = std::sqrt((float)(GRID_WIDTH * GRID_WIDTH + GRID_HEIGHT * GRID_HEIGHT));
        for (int y = Y_MIN; y <= Y_MAX; y++) {
            for (int x = X_MIN; x <= X_MAX; x++) {
                int idx = (y - Y_MIN) * GRID_WIDTH + (x - X_MIN);
                float dx = x - nest_x;
                float dy = y - nest_y;
                float dist = std::sqrt(dx * dx + dy * dy);
                h_grid[idx].nest_pheromone = 1.0f - dist / max_distance;
            }
        }
        
        // Place food sources - closer to nest
        std::vector<std::pair<int, int>> food_locations;
        total_food_available = 0;
        
        for (int i = 0; i < NUM_FOOD_SOURCES; i++) {
            int best_x = 0, best_y = 0;
            float best_score = -1;
            
            for (int attempt = 0; attempt < 50; attempt++) {
                // Place food at radius 30-80 from nest
                float angle = dist(rng) * 2 * M_PI;
                float radius = 30.0f + dist(rng) * 50.0f;
                int fx = (int)(nest_x + radius * std::cos(angle));
                int fy = (int)(nest_y + radius * std::sin(angle));
                
                // Clamp to bounds
                fx = std::max(X_MIN + 5, std::min(X_MAX - 5, fx));
                fy = std::max(Y_MIN + 5, std::min(Y_MAX - 5, fy));
                
                float score = 0;
                
                for (auto& loc : food_locations) {
                    float ddx = fx - loc.first;
                    float ddy = fy - loc.second;
                    score += ddx * ddx + ddy * ddy;
                }
                
                if (food_locations.empty()) score = 1;
                
                if (score > best_score) {
                    best_score = score;
                    best_x = fx;
                    best_y = fy;
                }
            }
            
            int idx = (best_y - Y_MIN) * GRID_WIDTH + (best_x - X_MIN);
            h_grid[idx].type = CELL_FOOD;
            h_grid[idx].food = FOOD_AMOUNT;
            food_locations.push_back({best_x, best_y});
            total_food_available += FOOD_AMOUNT;
        }
        
        // Spawn ants
        int ant_idx = 0;
        for (int ring = 1; ant_idx < NUM_ANTS && ring < 20; ring++) {
            for (int dx = -ring; dx <= ring && ant_idx < NUM_ANTS; dx++) {
                for (int dy = -ring; dy <= ring && ant_idx < NUM_ANTS; dy++) {
                    if (std::abs(dx) != ring && std::abs(dy) != ring) continue;
                    
                    int ax = nest_left_x + dx;
                    int ay = nest_left_y + dy;
                    
                    if (ax < X_MIN || ax > X_MAX || ay < Y_MIN || ay > Y_MAX) continue;
                    
                    int cell_idx = (ay - Y_MIN) * GRID_WIDTH + (ax - X_MIN);
                    if (h_grid[cell_idx].type == CELL_NEST) continue;
                    if (h_grid[cell_idx].type == CELL_FOOD) continue;
                    if (h_grid[cell_idx].ant_id != -1) continue;
                    
                    h_ants[ant_idx].x = ax;
                    h_ants[ant_idx].y = ay;
                    h_ants[ant_idx].state = ANT_SEARCHING;
                    h_ants[ant_idx].has_food = 0;
                    h_ants[ant_idx].orientation = dist(rng) * 2 * M_PI;
                    h_ants[ant_idx].new_dir = -1;
                    h_grid[cell_idx].ant_id = ant_idx;
                    ant_idx++;
                }
            }
        }
        
        // Copy to device
        cudaMemcpy(d_grid, h_grid.data(), GRID_SIZE * sizeof(CellGPU), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ants, h_ants.data(), NUM_ANTS * sizeof(AntGPU), cudaMemcpyHostToDevice);
        
        int zero = 0;
        cudaMemcpy(d_food_collected, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Initialize random states
        int blocks = (NUM_ANTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_curand_kernel<<<blocks, BLOCK_SIZE>>>(d_rand_states, seed);
        cudaDeviceSynchronize();
        
        total_food_collected = 0;
        ticks = 0;
    }
    
    void tick() {
        ticks++;
        
        // Phase 1: Decay pheromones
        int grid_blocks = (GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        decay_pheromones_kernel<<<grid_blocks, BLOCK_SIZE>>>(d_grid);
        
        // Phase 2: Find moves for all ants
        int ant_blocks = (NUM_ANTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        find_moves_kernel<<<ant_blocks, BLOCK_SIZE>>>(d_grid, d_ants, d_rand_states, nest_x, nest_y);
        
        // Phase 3: Apply moves (single thread for correctness)
        apply_moves_kernel<<<1, 1>>>(d_grid, d_ants, d_rand_states, d_food_collected, nest_x, nest_y);
        
        cudaDeviceSynchronize();
    }
    
    int get_food_collected() {
        cudaMemcpy(&total_food_collected, d_food_collected, sizeof(int), cudaMemcpyDeviceToHost);
        return total_food_collected;
    }
    
    bool is_complete() {
        return get_food_collected() >= total_food_available;
    }
    
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();
                
        while (!is_complete() && ticks < MAX_TICKS) {
            tick();
            
            if (ticks % 1000 == 0) {
                int collected = get_food_collected();
                std::cout << "Tick " << ticks 
                          << ": Collected " << collected 
                          << "/" << total_food_available << " food" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        int final_collected = get_food_collected();
        
        std::cout << "\n=== Simulation Complete (CUDA Version) ===" << std::endl;
        std::cout << "Total ticks: " << ticks << std::endl;
        std::cout << "Food collected: " << final_collected 
                  << "/" << total_food_available << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Time per tick: " << (double)duration.count() / ticks << " ms" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    unsigned seed = 42;
    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }
    
    // Print CUDA device info
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "=== Ant Colony Simulation (CUDA Version) ===" << std::endl;
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Grid size: " << GRID_WIDTH << " x " << GRID_HEIGHT << std::endl;
    std::cout << "Number of ants: " << NUM_ANTS << std::endl;
    std::cout << "Number of food sources: " << NUM_FOOD_SOURCES << std::endl;
    std::cout << "Food per source: " << FOOD_AMOUNT << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    AntSimulationCUDA simulation(seed);
    simulation.run();
    
    return 0;
}
