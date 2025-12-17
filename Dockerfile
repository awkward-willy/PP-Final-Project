FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    doctest-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Create build directory
RUN mkdir -p build

# Configure and build
WORKDIR /app/build
RUN cmake .. && make simulant simulant_headless

# Default command
CMD ["./simulant_headless"]
