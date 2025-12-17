#include "controllers/game.hpp"
#include <chrono>
#include <iostream>
#include "controllers/random.hpp"

#ifdef HEADLESS_MODE
Game::Game(const GameOptions& options) : grid(options.colonies_amount), headless(true), colonies_amount(options.colonies_amount) {
}
#else
Game::Game(const GameOptions& options) : view(options.fullscreen_view, options.width, options.height), grid(options.colonies_amount), headless(options.headless), colonies_amount(options.colonies_amount) {
}
#endif

void Game::restart() {
    grid.clear();
    grid = Grid(colonies_amount);
    delta.clear();
#ifndef HEADLESS_MODE
    if (!headless) {
        view.clear();
        view.init_grid(grid);
        view.renderAll();
    }
#endif
}

void Game::start() {
    size_t current_tick = 0;
    std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

#ifdef HEADLESS_MODE
    std::cout << "Starting simulation in headless mode..." << std::endl;
    while (grid.count_sugar() > 0) {
        loop(++current_tick);
    }
    std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    unsigned long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Simulation finished in " << duration << "ms (" << current_tick << " ticks)" << std::endl;
    return;
#else
    if (headless) {
        std::cout << "Starting simulation in headless mode..." << std::endl;
        while (grid.count_sugar() > 0) {
            loop(++current_tick);
        }
        std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
        unsigned long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "Simulation finished in " << duration << "ms (" << current_tick << " ticks)" << std::endl;
        return;
    }

    unsigned long minimal_delay = 2e8;  // one second
    std::chrono::high_resolution_clock::time_point previousTime = std::chrono::high_resolution_clock::now();
    view.init_grid(grid);
    view.renderAll();
    Event event;

    while ((event = view.event_manager()) != Event::close_request) {
        if (event == Event::restart)
            restart();

        unsigned long delay = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - previousTime).count();
        if (delay < minimal_delay) {
            view.update(delay / (double)minimal_delay, grid, current_tick);
            view.renderAll();
            continue;
        }
        loop(++current_tick);
        view.update_map(delta);
        previousTime = std::chrono::high_resolution_clock::now();

        if (grid.count_sugar() == 0) {
            std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
            unsigned long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            std::cout << "Simulation finished in " << duration << "ms (" << current_tick << " ticks)" << std::endl;
            std::cout << "All food collected. Exiting..." << std::endl;
            break;
        }
    }
#endif
}

void Game::loop(size_t current_tick) {
    delta.clear();

    // spawn new ants
    grid.map_colony([&](Colony* colony) {
        std::vector<Coordinates> new_ants = spawn_list[colony];
        std::vector<Coordinates> updated_new_ants;
        for (Coordinates coordinates : new_ants) {
            if (grid.get_cell(coordinates)->is_void())
                grid.spawn_ant(colony, coordinates);
            else
                updated_new_ants.push_back(coordinates);
        }
        spawn_list[colony] = updated_new_ants;
    });

    // perform tour logic
    std::vector<Ant*> killed;
    std::vector<Ant*> in_fight;
    grid.map_ants([&](Ant* ant) {
        if (std::find(killed.begin(), killed.end(), ant) == killed.end() && std::find(in_fight.begin(), in_fight.end(), ant) == in_fight.end())
            apply_ant_logic(ant, ant->find_move(grid, current_tick), killed, in_fight);
        else
            display_not_moving_ant(ant);
    });

    // kill ants
    for (Ant* ant : killed) {
        if (ant == NULL)
            continue;
        Action action;
        action.type = ActionType::AntDeath;
        action.updated.push_back(ant->get_location());
        // winner's colony
        action.colony = grid.get_cell(ant->get_location())->get_ant()->get_colony();
        delta.push_back(action);
        Colony* colony = ant->get_colony();
        colony->remove_ant(grid, colony->find_ant_index(ant));
    }
}

void Game::apply_ant_logic(Ant* ant, Cell* next_cell, std::vector<Ant*>& killed, std::vector<Ant*>& in_fight) {
    if (next_cell == NULL) {
        display_not_moving_ant(ant);
        return;
    }

    // fight
    if (next_cell->has_ant()) {
        // me move the attacker to the new case on the screen
        move_ant_on_view(ant, next_cell->get_location());

        if (flip_a_coin()) {
            killed.push_back(next_cell->get_ant());
            ant->move(grid, next_cell);
        } else {
            killed.push_back(ant);
            in_fight.push_back(next_cell->get_ant());
            grid.get_cell(ant->get_location())->set_ant(NULL);
            ant->set_location(next_cell->get_location());
        }
    }

    // let's take sugar
    else if (!ant->has_sugar() && next_cell->has_sugar()) {
        ant->add_sugar();
        next_cell->remove_sugar();
        move_ant_on_view(ant, ant->get_location());
    }

    // let's deposit sugar
    else if (next_cell->is_nest() && next_cell->get_nest() == ant->get_colony()) {
        if (ant->has_sugar()) {
            grid.get_cell(ant->get_location())->add_sugar_pheromon();
            if (ant->deposit_sugar())  // if an ant spawned
                spawn_list[ant->get_colony()].push_back(ant->get_location());
        }
        move_ant_on_view(ant, ant->get_location());
    }

    // juste move
    else {
        if (ant->has_sugar())
            grid.get_cell(ant->get_location())->add_sugar_pheromon();

        move_ant_on_view(ant, next_cell->get_location());
        ant->move(grid, next_cell);
    }
}

void Game::display_not_moving_ant(Ant* ant) {
    move_ant_on_view(ant, ant->get_location());
}

void Game::move_ant_on_view(Ant* ant, Coordinates location) {
    Action action;
    action.type = ActionType::AntMove;
    action.updated.push_back(ant->get_location());
    action.updated.push_back(location);
    action.colony = ant->get_colony();
    delta.push_back(action);
}