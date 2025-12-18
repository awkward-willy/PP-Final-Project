# Ant Colony Simulation - Parallel Programming Final Project

這是一個螞蟻群落模擬程式，模擬螞蟻尋找食物並將其搬回巢穴的行為。

## 模擬設定

- **網格大小**: 256 x 256
- **螞蟻數量**: 500 隻 (單一群落)
- **食物點數量**: 10 個
- **每個食物點的食物量**: 20 單位
- **總食物量**: 200 單位

## 版本說明

### 1. CPU 版本 (Sequential)

純 C++ 實作，單執行緒順序執行。

### 2. OpenMP 版本 (Parallel)

使用 OpenMP 進行平行化處理：

- 費洛蒙衰減平行化
- 螞蟻移動決策平行化
- 使用 thread-local RNG 避免競爭

### 3. CUDA 版本 (GPU)

使用 CUDA 進行 GPU 加速：

- 費洛蒙衰減在 GPU 上平行執行
- 螞蟻移動決策在 GPU 上平行執行
- 使用 cuRAND 進行隨機數生成

## 編譯方式

### 使用 Makefile (推薦)

```bash
# 編譯 CPU 版本
make cpu

# 編譯 OpenMP 版本
make openmp

# 編譯 CUDA 版本 (需要 CUDA toolkit)
make cuda

# 編譯所有版本 (CPU + OpenMP)
make all

# 清理編譯產物
make clean
```

### 使用 CMake

```bash
mkdir build && cd build

# 基本編譯 (CPU + OpenMP)
cmake -DCMAKE_BUILD_TYPE=Release ..

# 啟用 CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON ..

# 編譯
make -j$(nproc)
```

## 執行方式

```bash
# CPU 版本
./build/ant_cpu [random_seed]

# OpenMP 版本
./build/ant_openmp [random_seed] [num_threads]

# CUDA 版本
./build/ant_cuda [random_seed]
```

### 範例

```bash
# 使用預設種子 42 執行 CPU 版本
./build/ant_cpu

# 使用種子 123 執行 OpenMP 版本，4 個執行緒
./build/ant_openmp 123 4

# 執行 CUDA 版本
./build/ant_cuda 42
```

## 輸出說明

程式會輸出：

1. 模擬配置資訊
2. 每 1000 個 tick 的進度報告
3. 最終統計資訊：
   - 總 tick 數
   - 收集的食物數量
   - **執行時間 (毫秒)**
   - 每個 tick 的平均時間

## 執行效能測試

```bash
# 執行 CPU 和 OpenMP 的效能測試
make benchmark

# 執行 CUDA 的效能測試
make benchmark-cuda
```

## 檔案結構

```
src_headless/
├── common/
│   ├── config.hpp      # 共用配置常數
│   └── types.hpp       # 共用資料結構
├── cpu/
│   └── main.cpp        # CPU 版本實作
├── openmp/
│   └── main.cpp        # OpenMP 版本實作
└── cuda/
    ├── config_cuda.cuh # CUDA 配置
    └── main.cu         # CUDA 版本實作
```

## 系統需求

- C++17 相容編譯器 (g++ 7+ 或 clang++ 5+)
- OpenMP 支援 (通常 GCC 內建)
- CUDA Toolkit 10.0+ (僅 CUDA 版本需要)

## 演算法說明

### 螞蟻行為

1. **尋找食物模式**：

   - 跟隨食物費洛蒙
   - 傾向遠離巢穴移動
   - 保持移動方向的一致性
   - 找到食物後切換為返回模式

2. **返回巢穴模式**：
   - 跟隨巢穴費洛蒙
   - 沿途留下食物費洛蒙
   - 到達巢穴後放下食物並切換回尋找模式

### 費洛蒙機制

- 巢穴費洛蒙：基於距離預先計算，不會衰減
- 食物費洛蒙：螞蟻返回時留下，每個 tick 以 0.975 的速率衰減

## 授權

MIT License
