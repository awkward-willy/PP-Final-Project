#pragma once

#include "controllers/cli.hpp"
#include "models/action.hpp"
#include "models/grid.hpp"
#ifndef HEADLESS_MODE
#include "view/view.hpp"
#endif

class Game {
   public:
    /**
     * to intialize a Game object
     *
     * @param const GameOptions& game options
     *
     * @confidence 5: no tests needed
     * @author Thomas Marchand
     **/
    Game(const GameOptions& options);

    /**
     * to start the game
     *
     * @confidence 3: no unit test possible
     * @author Thomas Marchand
     **/
    void start();

    /**
     * to restart the game
     *
     * @confidence 3: no unit test possible
     * @author Frédéric Becerril
     **/
    void restart();

   private:
#ifndef HEADLESS_MODE
    View view;
#endif
    Grid grid;
    bool headless;
    size_t colonies_amount;
    std::vector<Action> delta;
    std::map<Colony*, std::vector<Coordinates>> spawn_list;

    /**
     * the game loop called every tick
     *
     * @param size_t the current tick
     *
     * @confidence 4
     * @author Thomas Marchand
     **/
    void loop(size_t current_tick);

    /**
     * the logic applied to an ant at every tick
     *
     * @param Ant* a pointer to the ant
     * @param Cell* a pointer to its next location (or NULL if it does not move)
     * @param std::vector<Ant *>& a reference to the killed ants vector
     * @param std::vector<Ant *>a reference to the "in fight" ants vector
     *
     * @confidence 4
     * @author Thomas Marchand
     **/
    void apply_ant_logic(Ant* ant, Cell* next_cell, std::vector<Ant*>& killed, std::vector<Ant*>& in_fight);

    /**
     * to display a static ant
     *
     * @param Ant* a pointer to the ant to display
     *
     * @confidence 3: no unit test possible
     * @author Thomas Marchand
     **/
    void display_not_moving_ant(Ant* ant);

    /**
     * to display a moving ant
     *
     * @param Ant* a pointer to the ant to display
     * Coordinates its destination
     *
     * @confidence 3: no unit test possible
     * @author Thomas Marchand and Frédéric Becerril
     **/
    void move_ant_on_view(Ant* ant, Coordinates location);
};