#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_timer.h>
#include <algorithm>
#include <map>
#include <unordered_set>

#include "models/action.hpp"
#include "models/grid.hpp"
#include "view/color.hpp"
#include "view/hud.hpp"

enum struct Event {
    close_request,
    restart,
    none
};

class View {
   public:
    View(bool fullScreen, int w = 800, int h = 800);
    View(int w, int h);
    ~View();

    void init_grid();
    void init_hud();
    Event event_manager();

    void clear();

    void draw_cell_rect(const Coordinates& c, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool fill = true);
    void draw_cell_rect(double x_rect, double y_rect, uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool fill = true);

    void draw_cell_circle(const Coordinates& c, uint8_t r, uint8_t g, uint8_t b, uint8_t a, double scale = 0.25);
    void draw_cell_circle(double x_rect, double y_rect, uint8_t r, uint8_t g, uint8_t b, uint8_t a, double scale = 0.25);

    void init_grid(const Grid& grid);
    void init_entities(const Grid& grid);
    void update_pheromones(const Grid& grid, size_t current_block);
    void update_entities(const Grid& grid, double delta_time);
    void update_grid();
    void update_info(const Grid& grid);

    void update(double time, const Grid& grid, size_t current_block);
    void update_map(std::vector<Action> d);

    void renderAll();

   private:
    SDL_Window* window;
    SDL_Renderer* render;
    int window_w;
    int window_h;
    int cell_size;
    int cell_w;
    int cell_h;
    int grid_w;
    int grid_h;
    int grid_x;
    int grid_y;

    SDL_Rect top;
    SDL_Rect right;
    SDL_Rect left;
    SDL_Rect down;

    bool clicked;
    bool double_clicked;
    int scroll;
    int mouse_x;
    int mouse_y;

    double new_w;
    double new_h;
    double x_shift;
    double y_shift;

    int x_start;
    int x_end;
    int y_start;
    int y_end;

    SDL_Texture* grid_texture;

    SDL_Texture* background_texture;
    SDL_Texture* entities_texture;
    SDL_Texture* pheromones_texture;
    SDL_Texture* info_texture;

    TTF_Font* font;

    std::map<const Colony*, rgb> m;
    std::map<const Colony*, bool> disp_pheromones;

    std::vector<Action> delta;

    bool end;
    bool restart;

    HUD hud;

    SDL_Rect texture_rect;
    SDL_Rect dest_rect;
};

void DrawCircle(SDL_Renderer* renderer, int32_t centreX, int32_t centreY, int32_t radius);
void SDL_RenderFillCircle(SDL_Renderer* renderer, int x, int y, int radius);