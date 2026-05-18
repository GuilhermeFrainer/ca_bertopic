# ==============================================================================
# STAGE 1: Build Environment (Builder)
# ==============================================================================
# We use the devel image to provide the compilers and headers needed to build 
# Python and R packages from source (especially for Arrow, RAPIDS, and STM).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
# Ensure R arrow package downloads/builds its own C++ binaries
ENV NOT_CRAN=true
ENV LIBARROW_MINIMAL=false

# 1. Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    ca-certificates \
    gnupg \
    build-essential \
    cmake \
    git \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | tee /etc/apt/sources.list.d/cran.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
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
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv' for Python dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

# 3. Install Python virtual environment
# We install dependencies first and immediately clean the cache to save space
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev && uv cache clean

# 4. Install R library via renv
# We copy only configuration files to maintain environment integrity
COPY renv.lock .Rprofile ./
COPY renv/activate.R renv/settings.json renv/
# Restore R packages (this will compile them for Linux)
RUN R -e "install.packages('renv', repos='https://cloud.r-project.org')" \
    && R -e "renv::restore()" \
    # Remove the global renv cache which can be several gigabytes
    && rm -rf /root/.local/share/renv

# ==============================================================================
# STAGE 2: Runtime Environment (Final)
# ==============================================================================
# We switch to a runtime-only image which lacks compilers and development headers, 
# significantly reducing the final image footprint.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
# Add the Python virtual environment to the system path
ENV PATH="/app/.venv/bin:$PATH"

# 1. Install runtime-only system libraries and setup repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    ca-certificates \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | tee /etc/apt/sources.list.d/cran.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    r-base \
    libcurl4 \
    libxml2 \
    libfontconfig1 \
    libharfbuzz0b \
    libfribidi0 \
    libfreetype6 \
    libpng16-16 \
    libtiff5 \
    libjpeg8 \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the pre-built environments from the builder stage
# This copies only the finalized binaries and libraries, not the build artifacts.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/renv/library /app/renv/library
COPY --from=builder /app/renv/activate.R /app/renv/settings.json /app/renv/
COPY .Rprofile ./

# 3. Copy application source code (respecting .dockerignore)
# This includes scripts/, src/, and experiments/.
COPY . .

# Final verification of the environment
RUN python --version && R --version

# Default command for the container
CMD ["bash"]
