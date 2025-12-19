# Ant Colony Simulation with OpenMP Visualization

這是一個使用 OpenMP 平行化的螞蟻群落模擬程式，具有 SDL2 即時視覺化介面。

## 功能特點

- **256x256 網格**
- **500 隻螞蟻**（單一群落）
- **10 個食物來源**
- **OpenMP 平行化**：費洛蒙衰減和螞蟻決策
- **即時視覺化**：使用 SDL2 渲染
- **互動控制**：暫停、重置、調整速度

## 快速開始 (Docker)

使用 Docker 是最簡單的方式，不需要安裝任何本地依賴。

### 前置要求

- Docker 和 docker-compose
- X11 顯示服務器 (Linux)

### 啟動

```bash
cd src_visualization

# 允許 Docker 存取 X11 顯示
xhost +local:

# 使用 docker-compose 啟動
docker-compose up ant-visualization

# 或使用 8 個執行緒
OMP_NUM_THREADS=8 docker-compose up ant-visualization
```

### 停止

按 `Q` 或 `ESC` 鍵關閉視窗，或使用：

```bash
docker-compose down
```

## 本地編譯（可選）

如果您想在本地編譯，需要安裝以下依賴：

### Ubuntu/Debian 安裝相依套件

```bash
sudo apt-get install build-essential libsdl2-dev libsdl2-ttf-dev fonts-dejavu-core
```

### Arch Linux

```bash
sudo pacman -S base-devel sdl2 sdl2_ttf ttf-dejavu
```

### NixOS / Nix

```bash
nix-shell -p gcc gnumake SDL2 SDL2_ttf
```

## 編譯與執行

### 使用 Makefile

```bash
cd src_visualization
make
./ant_visualization
```

### 指定執行緒數量

```bash
OMP_NUM_THREADS=8 ./ant_visualization
# 或
./ant_visualization 8
```

### 使用 CMake

```bash
cd src_visualization
mkdir build && cd build
cmake ..
make
./ant_visualization
```

## Docker 使用

### 前置作業（允許 X11 存取）

在 Linux 主機上執行：

```bash
xhost +local:docker
```

### 使用 docker-compose 啟動

```bash
cd src_visualization

# 建置並啟動（預設 4 執行緒）
docker-compose up --build

# 指定執行緒數量
OMP_NUM_THREADS=8 docker-compose up --build

# 使用 8 執行緒的設定檔
docker-compose --profile 8threads up ant-visualization-8threads --build
```

### 手動 Docker 指令

```bash
# 建置 Docker 映像
docker build -t ant-visualization .

# 執行（需要 X11 轉發）
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    --network host \
    ant-visualization
```

## 操作說明

| 按鍵        | 功能            |
| ----------- | --------------- |
| `SPACE`     | 暫停/繼續模擬   |
| `R`         | 重置模擬        |
| `↑`         | 加快速度 (2x)   |
| `↓`         | 減慢速度 (0.5x) |
| `Q` / `ESC` | 結束程式        |

## 視覺化說明

- **白色點**：沒有攜帶食物的螞蟻
- **黃色點**：攜帶食物的螞蟻
- **棕色區域**：巢穴
- **綠色區域**：食物來源
- **紅色軌跡**：食物費洛蒙（指向食物）
- **藍色軌跡**：家費洛蒙（指向巢穴）

## 檔案結構

```
src_visualization/
├── main.cpp          # 主程式（模擬邏輯 + 視覺化）
├── config.hpp        # 設定常數
├── types.hpp         # 資料結構定義
├── color.hpp         # 顏色工具
├── Makefile          # 編譯腳本
├── CMakeLists.txt    # CMake 設定
├── Dockerfile        # Docker 映像定義
├── docker-compose.yml # Docker Compose 設定
└── README.md         # 本文件
```

## 效能說明

此版本使用 OpenMP 平行化以下操作：

1. **費洛蒙衰減**：所有格子的費洛蒙同時衰減
2. **螞蟻決策**：每隻螞蟻的下一步移動決策並行計算

螞蟻實際移動仍是循序執行，以確保正確性（避免競爭條件）。

典型效能提升（相比單執行緒）：

- 4 執行緒：約 2-2.5x 加速
- 8 執行緒：約 3-4x 加速

## 故障排除

### Docker 無法顯示畫面

1. 確認已執行 `xhost +local:docker`
2. 確認 DISPLAY 環境變數正確設定
3. 嘗試使用 `--network host` 模式

### 編譯錯誤：找不到 SDL2

確認已安裝 SDL2 開發套件：

```bash
pkg-config --libs sdl2 SDL2_ttf
```

### 字型無法載入

程式會嘗試從多個路徑載入 DejaVu 字型。如果 HUD 不顯示文字，安裝字型套件：

```bash
sudo apt-get install fonts-dejavu-core
```
