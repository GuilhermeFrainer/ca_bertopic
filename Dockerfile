# Use a devel image to ensure we have headers for compiling R/Python packages
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# 1. Install system dependencies
# We need python3.12, R, and build-essential for compiling various packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libarrow-dev \
    git \
    curl \
    rsync \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv' for Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Set up the working directory
WORKDIR /app

# 4. Install Python dependencies
# We copy only the files needed for installation first to leverage layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 5. Install R dependencies via renv
# We copy ONLY the configuration files, NOT the library/ folder
COPY renv.lock .Rprofile ./
COPY renv/activate.R renv/settings.json renv/
# Restore the R environment (this will compile packages for Linux)
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')"
RUN R -e "renv::restore()"

# 6. Copy the rest of the application (respecting .dockerignore)
# This includes src/, scripts/, experiments/, tests/, etc.
COPY . .

# Final check of the environment
RUN uv run python --version && R --version

# Default command
CMD ["bash"]
