#include "view/view.hpp"
#include <algorithm>

double scale_high_dpi = 0;

View::View(bool fullScreen, int width, int height) {
    int w;
    int h;

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);
    if (!fullScreen)
        window = SDL_CreateWindow("Ant", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_ALLOW_HIGHDPI);
    else
        window = SDL_CreateWindow("Ant", 0, 0, 0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_ALLOW_HIGHDPI);
    render = SDL_CreateRenderer(window, -1, SDL_RENDERER_TARGETTEXTURE | SDL_RENDERER_ACCELERATED);
    if (!render) {
        render = SDL_CreateRenderer(window, -1, SDL_RENDERER_TARGETTEXTURE);
    }
    SDL_SetRenderDrawBlendMode(render, SDL_BLENDMODE_BLEND);

    SDL_GL_GetDrawableSize(window, &window_w, &window_h);
    SDL_GetWindowSize(window, &w, &h);

    scale_high_dpi = window_w / (double)w;

    init_grid();
    init_hud();

    end = false;
    restart = false;
}

View::View(int w, int h) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);

    TTF_Init();

    window = SDL_CreateWindow("Ant", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_ALLOW_HIGHDPI);
    render = SDL_CreateRenderer(window, -1, SDL_RENDERER_TARGETTEXTURE | SDL_RENDERER_ACCELERATED);
    if (!render) {
        render = SDL_CreateRenderer(window, -1, SDL_RENDERER_TARGETTEXTURE);
    }

    SDL_SetRenderDrawBlendMode(render, SDL_BLENDMODE_BLEND);
    SDL_GL_GetDrawableSize(window, &window_w, &window_h);

    scale_high_dpi = window_w / (double)w;

    // SDL_Surface *surface = SDL_GetWindowSurface(window); || might be usefull to record the game

    init_grid();
    init_hud();

    end = false;
    restart = false;
}

View::~View() {
    SDL_DestroyRenderer(render);
    SDL_DestroyWindow(window);

    m.clear();
    disp_pheromones.clear();
    SDL_DestroyTexture(background_texture);
    SDL_DestroyTexture(entities_texture);
    SDL_DestroyTexture(pheromones_texture);
    SDL_DestroyTexture(info_texture);
    TTF_CloseFont(font);

    TTF_Quit();
    SDL_Quit();
}

void View::init_grid() {
    cell_size = std::min(window_w / SPACE_WIDTH, window_h / SPACE_HEIGHT);

    cell_w = cell_size;
    cell_h = cell_size;

    grid_w = cell_w * SPACE_WIDTH;
    grid_h = cell_h * SPACE_HEIGHT;
    new_w = cell_size;
    new_h = cell_size;

    x_shift = 0;
    y_shift = 0;
    if (grid_w < window_w) {
        x_shift = ((window_w - grid_w) / 2) / cell_size;
    }
    if (grid_h < window_h) {
        y_shift = ((window_h - grid_h) / 2) / cell_size;
    }
}

void View::init_hud() {
    hud.init_hud(window, render);

    hud.create_menu("Pause");
    hud.add_rect_draw("Pause", "", {0.4, 0.35, 0.2, 0.28}, {0x1C, 0x1F, 0x2A, 150}, "ressources/Marianne-Regular.otf", 42);

    // hud.add_rect_draw("Pause", "premier test", {0,0,0.10,0.10}, {0xFF,0xFF,0xFF,0xFF}, "ressources/Marianne-Regular.otf", 42);
    hud.add_button("Pause", "Resume", {0.45, 0.38, 0.1, 0.06}, {0x97, 0xF0, 0xCC, 0xFF}, "ressources/Marianne-Regular.otf", 21,
                   [&]() {
                       hud.hide_menu("Pause");
                   });
    hud.add_button("Pause", "Restart", {0.45, 0.46, 0.1, 0.06}, {0xF1, 0xDB, 0x96, 0xFF}, "ressources/Marianne-Regular.otf", 21,
                   [&]() {
                       restart = true;
                   });
    hud.add_button("Pause", "Exit", {0.45, 0.54, 0.1, 0.06}, {0xF1, 0x96, 0xA6, 0xFF}, "ressources/Marianne-Regular.otf", 21,
                   [&]() {
                       end = true;
                   });

    hud.hide_menu("Pause");
}

Event View::event_manager() {
    SDL_Event event;
    clicked = false;
    double_clicked = false;
    scroll = 0;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                return Event::close_request;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        if (hud.menu_is_hidden("Pause")) {
                            hud.show_menu("Pause");
                        } else {
                            hud.hide_menu("Pause");
                        }
                        break;
                    case SDLK_UP:
                        y_shift += 1;
                        break;
                    case SDLK_DOWN:
                        y_shift -= 1;
                        break;
                    case SDLK_RIGHT:
                        x_shift -= 1;
                        break;
                    case SDLK_LEFT:
                        x_shift += 1;
                        break;
                    default:
                        break;
                }
                break;

            case SDL_MOUSEBUTTONDOWN:
                mouse_x = event.button.x;
                mouse_y = event.button.y;
                if (event.button.clicks == 1) {
                    clicked = true;
                } else {
                    double_clicked = true;
                }
                break;
            case SDL_MOUSEWHEEL:
                SDL_GetMouseState(&mouse_x, &mouse_y);
                if (event.wheel.y > 0) {
                    scroll = -1;
                } else if (event.wheel.y < 0) {
                    scroll = 1;
                }
                break;
            default:
                break;
        }
    }
    if (end)
        return Event::close_request;
    if (restart) {
        restart = false;
        return Event::restart;
    }
    return Event::none;
}

void View::init_grid(const Grid& grid) {
    // background_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, texture_rect.w, texture_rect.h);
    // SDL_SetTextureBlendMode(background_texture, SDL_BLENDMODE_BLEND);
    // entities_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, texture_rect.w, texture_rect.h);
    // SDL_SetTextureBlendMode(entities_texture, SDL_BLENDMODE_BLEND);
    // pheromones_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, texture_rect.w, texture_rect.h);
    // SDL_SetTextureBlendMode(pheromones_texture, SDL_BLENDMODE_ADD);

    background_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, window_w, window_h);
    SDL_SetTextureBlendMode(background_texture, SDL_BLENDMODE_BLEND);
    entities_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, window_w, window_h);
    SDL_SetTextureBlendMode(entities_texture, SDL_BLENDMODE_BLEND);
    pheromones_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, window_w, window_h);
    SDL_SetTextureBlendMode(pheromones_texture, SDL_BLENDMODE_ADD);
    info_texture = SDL_CreateTexture(render, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, window_w, window_h);
    SDL_SetTextureBlendMode(info_texture, SDL_BLENDMODE_BLEND);

    font = TTF_OpenFont("ressources/Marianne-Regular.otf", 32);

    double base_tint = create_base_tint();

    for (size_t i = 0; i < grid.colonies.size(); i++) {
        m[grid.colonies[i]] = get_tint(i, grid.colonies.size(), base_tint, 0.35, 0.93);
        disp_pheromones[grid.colonies[i]] = false;
    }

    SDL_SetRenderTarget(render, background_texture);
    SDL_SetRenderDrawColor(render, 0, 0, 0, 0);
    SDL_RenderClear(render);
    SDL_SetRenderDrawColor(render, 0x2C, 0x3A, 0x47, 0xFF);
    // SDL_RenderSetScale(render, 2, 2);
    for (int i = 0; i < SPACE_HEIGHT; i++) {
        for (int j = 0; j < SPACE_WIDTH; j++) {
            SDL_Rect rect = {j * cell_w, i * cell_h, cell_w, cell_h};
            SDL_RenderDrawRect(render, &rect);
        }
    }
    // SDL_RenderSetScale(render, 1, 1);
    SDL_SetRenderTarget(render, NULL);
    init_entities(grid);
    update_pheromones(grid, 0);
}

void View::init_entities(const Grid& grid) {
    SDL_SetRenderTarget(render, entities_texture);

    SDL_SetRenderDrawColor(render, 0, 0, 0, 0);
    SDL_RenderClear(render);

    for (Cell* cell : grid.map) {
        if (cell->is_nest()) {
            rgb color = m[cell->get_nest()];
            draw_cell_rect(cell->get_location(), color.r, color.g, color.b, color.a);
        }
        if (cell->has_ant()) {
            Ant* a = cell->get_ant();
            rgb color = m[a->get_colony()];
            draw_cell_circle(cell->get_location(), color.r, color.g, color.b, color.a);
        }
        if (cell->has_sugar()) {
            draw_cell_rect(cell->get_location(), 0xFF, 0xFF, 0xFF, 0xFF);

            //(c, 0xFF, 0xFF, 0xFF);
        }
    }
    SDL_SetRenderTarget(render, NULL);
}

void View::clear() {
    m.clear();
    disp_pheromones.clear();
    delta.clear();
    SDL_DestroyTexture(background_texture);
    SDL_DestroyTexture(entities_texture);
    SDL_DestroyTexture(pheromones_texture);
    SDL_DestroyTexture(info_texture);
    TTF_CloseFont(font);
}