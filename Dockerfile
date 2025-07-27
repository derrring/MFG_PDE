# Dockerfile for MFG_PDE computational framework
# Multi-stage build for security and size optimization

FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements and setup files
COPY pyproject.toml README.md ./
COPY mfg_pde/ ./mfg_pde/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir psutil colorlog hypothesis

# Production stage
FROM python:3.10-slim as production

# Set security-focused labels
LABEL maintainer="MFG_PDE Team" \
      description="Mean Field Games PDE solver framework" \
      version="0.1.0" \
      security.scan="enabled"

# Set environment variables for security
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    libopenmpi3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r mfguser && useradd -r -g mfguser -u 1001 mfguser

# Create application directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=mfguser:mfguser mfg_pde/ ./mfg_pde/
COPY --chown=mfguser:mfguser examples/ ./examples/
COPY --chown=mfguser:mfguser pyproject.toml README.md ./

# Install package in production mode
RUN pip install --no-cache-dir -e .

# Create directories for user
RUN mkdir -p /app/output /app/logs && \
    chown -R mfguser:mfguser /app

# Switch to non-root user
USER mfguser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import mfg_pde; print('MFG_PDE package imported successfully')" || exit 1

# Set working directory for user
WORKDIR /app

# Default command - run basic example
CMD ["python", "-c", "from mfg_pde import ExampleMFGProblem, create_fast_solver; print('MFG_PDE container ready')"]

# Expose port for potential web interface (if added later)
EXPOSE 8080

# Add volume for output data
VOLUME ["/app/output", "/app/logs"]