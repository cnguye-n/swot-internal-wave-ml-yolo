# syntax=docker/dockerfile:1

FROM mambaorg/micromamba:2.8.1

LABEL org.opencontainers.image.source=https://github.com/cnguye-n/swot-internal-wave-ml-yolo

# ------------------------------------------------------------
# SYSTEM DEPENDENCIES
# ------------------------------------------------------------
# Needed by OpenCV / Ultralytics in Linux containers.

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER


# ------------------------------------------------------------
# APPLICATION
# ------------------------------------------------------------

WORKDIR /app


# ------------------------------------------------------------
# PYTHON ENVIRONMENT
# ------------------------------------------------------------
# Copy environment file first so Docker can cache this expensive layer.

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /app/environment.yml

RUN micromamba create -y -f environment.yml && \
    micromamba clean --all --yes

ENV ENV_NAME=swot-internal-wave
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg


# ------------------------------------------------------------
# MODEL + API
# ------------------------------------------------------------

    COPY --chown=$MAMBA_USER:$MAMBA_USER \
    swot_internal_wave_detector.py \
    api.py \
    last.pt \
    /app/

COPY --chown=$MAMBA_USER:$MAMBA_USER \
    pipeline \
    /app/pipeline


# ------------------------------------------------------------
# FASTAPI / UVICORN
# ------------------------------------------------------------

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]