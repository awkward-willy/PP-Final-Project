# Ant Colony Simulation with OpenMP + SDL2 Visualization
# Multi-stage build for optimized image size

FROM ubuntu:22.04 AS builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    pkg-config \
    libsdl2-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy source files
COPY main.cpp Makefile config.hpp ./

# Build the application
RUN make

# Runtime stage
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies including X11 libs
RUN apt-get update && apt-get install -y \
    libsdl2-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libgomp1 \
    fonts-dejavu-core \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy built binary from builder stage
COPY --from=builder /app/ant_visualization /app/

# Set environment variables for SDL
ENV SDL_VIDEODRIVER=x11

# Default number of OpenMP threads
ENV OMP_NUM_THREADS=4

# Run the application
CMD ["./ant_visualization"]
