#pragma once

#include <sys/types.h>
#include <array>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>
#include "controllers/ant.hpp"
#include "models/space.hpp"

class Cell {
   public:
    /**
     * Constructor of a Cell object
     *
     * @param const Coordinates location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Cell(const Coordinates location);

    /**
     * getter for Cell location
     *
     * @return const Coordinates location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Coordinates get_location() const;

    /**
     * Check if a cell is empty
     *
     * @return true if void
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    bool is_void();

    /**
     * Check if a cell contains sugar
     *
     * @return true if cell has sugar
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    bool has_sugar();

    /**
     * Add sugar to a cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void add_sugar();

    /**
     * Remove sugar from a cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void remove_sugar();

    /**
     * Check if this cell is part of a nest
     *
     * return true if this cell is a nest
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    bool is_nest();

    /**
     * Set a nest associated to this cell
     *
     * @param Colony* a pointer to the colony nest
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void set_nest(Colony* colony);

    /**
     * A getter for the nest associated to this cell
     *
     * @return Colony* a pointer to the nest associated to this cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Colony* get_nest();

    /**
     * Check if an ant is on the cell
     *
     * @return true if an ant is on the cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    bool has_ant();

    /**
     * Set the ant associated to a cell
     *
     * @param Ant* a pointer to the tan
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void set_ant(Ant* ant);

    /**
     * Get the ant associated to the cell
     *
     * @return Ant* a pointer to the associated ant
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Ant* get_ant();

    /**
     * Get the nest pheromones associated to a Colony
     *
     * @param const Colony* a colony
     *
     * @return double the pheromones
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    double get_nest_pheromones(const Colony* colony);

    /**
     * Get the sugar pheromones on this cell
     *
     * @return double sugar pheromones
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    double get_sugar_pheromones();

    /**
     * Adds the sugar pheromones of the passage of an ant
     *
     * @return Ant* a pointer to the associated ant
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void add_sugar_pheromon();

    /**
     * Refresh cell pheromones to the current tick
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void update(size_t current_tick);

   private:
    Coordinates location;
    size_t sugar = 0;
    double sugar_pheromones = 0;
    std::map<const Colony*, double> nest_pheromones;
    Colony* nest = NULL;
    Ant* ant = NULL;
    size_t last_block = 0;

    friend class Grid;
    friend class Colony;
};

class Grid {
   public:
    /**
     * Grid constructor
     *
     * @param size_t amount of colonies to create
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Grid(size_t colonies_amount);

    template <class Function>
    /**
     * Apply a lambda function taking a Colony* parameter to all colonies of the grid
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void map_colony(Function&& function);

    template <class Function>
    /**
     * Apply a lambda function taking a Ant* parameter to all ants of the grid
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    void map_ants(Function&& function);

    /**
     * Get a cell at a specific location (0, 0) + vector
     *
     * @return Cell* a pointer to the cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Cell* get_cell(const Vector& vector) const;

    /**
     * Get a cell at a specific location
     *
     * @param int location on the x axis
     * @param int location on the y axis
     *
     * @return Cell* a pointer to the cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Cell* get_cell(int x, int y) const;

    /**
     * Get a cell at a specific location
     *
     * @param Coordinates location
     * @param size_t current_tick
     *
     * @return Cell* a pointer to the cell
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    Cell* get_cell(Coordinates location, size_t current_tick = 0) const;

    size_t count_sugar() const;

    /**
     * Spawn an ant from a specific colony to a specific location
     *
     * @param Colony* a pointer to the colony
     * @param Coordinates location
     *
     * @confidence 5
     * @author Thomas Marchand
     **/
    void spawn_ant(Colony* colony, Coordinates location);

    /**
     * Clear the grid and free the memory
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    void clear();

   private:
    std::vector<Cell*> map;
    std::vector<Colony*> colonies;

    /**
     * A getter for the nest cells
     *
     * @return std::array<Cell *, 4> nest cells
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    std::array<Cell*, 4> find_nest_cells();

    /**
     * Summon ants of a colony
     *
     * @param Colony* colony
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    void summon_ants(Colony* colony);

    /**
     * Summon sugar on the grid
     *
     * @param size_t amount of sugar to summon
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    void summon_sugars(size_t amount);

    /**
     * Find an empty cell on the grid
     *
     * @return Cell* a pointer to an empty cell
     *
     * @confidence 3
     * @author Thomas Marchand
     **/
    Cell* find_empty_cell();

    friend class View;
};

#include "models/grid.tpp"