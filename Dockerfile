# RazorShield AI — multi-stage build.
#
# The model, calibrator and seeded event store are generated, not committed, so
# a clone alone cannot serve traffic. The build stage runs the whole pipeline
# and the runtime stage copies only what serving actually needs: ~630 KB of
# artifacts plus the event store. The 20 MB of training CSVs never leave here.
#
# Building takes about four minutes and peaks near 400 MB. Serving settles
# around 270 MB, which fits a 512 MB free tier.

# --------------------------------------------------------------------------
# build: install, run the pipeline, produce artifacts
# --------------------------------------------------------------------------
FROM python:3.12-slim AS build

# libgomp1 is the OpenMP runtime LightGBM and XGBoost link against; without it
# the import fails at runtime with a bare "libgomp.so.1: cannot open shared
# object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY razorshield ./razorshield
RUN pip install --no-cache-dir .

# One deterministic pass. Same seed in, byte-identical artifacts out, so the
# image is reproducible from the source alone. The quality gate and the
# training/serving parity check both run here: if either fails, the build
# fails rather than shipping a model nobody verified.
RUN python -m razorshield.data.generate --out data/raw \
 && python -m razorshield.data.validate --data data/raw \
 && python -m razorshield.models.compare --data data/raw --out reports \
 && python -m razorshield.explain.run --data data/raw --out reports \
 && python -m razorshield.scoring.score --data data/raw --out reports \
 && python -m razorshield.serving.parity \
 && python -m razorshield.serving.seed --db data/razorshield.db --score 800

# --------------------------------------------------------------------------
# runtime: the server and nothing else
# --------------------------------------------------------------------------
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# The event store is written to on every scored payment, so it has to be owned
# by the runtime user rather than root.
COPY --from=build --chown=app:app /build/razorshield        ./razorshield
COPY --from=build --chown=app:app /build/models             ./models
COPY --from=build --chown=app:app /build/reports            ./reports
COPY --from=build --chown=app:app /build/data/raw/manifest.json ./data/raw/manifest.json
COPY --from=build --chown=app:app /build/data/razorshield.db    ./data/razorshield.db

USER app

# Hosts inject their own port. 8077 is only the local default.
ENV PORT=8077
EXPOSE 8077

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8077')+'/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "exec uvicorn razorshield.serving.app:app --host 0.0.0.0 --port ${PORT:-8077}"]
