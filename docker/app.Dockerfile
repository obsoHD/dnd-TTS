# Bag app image: the render pipeline (canon -> synth -> gate -> master), the SQLite
# store, the bag CLI and, from M2, the FastAPI service on :8020. CPU only: GPU 1
# belongs to the TTS container and speaker similarity (resemblyzer) costs 0.08 s/clip
# on CPU, so no CUDA, no torchaudio, no parselmouth ship in this image.
FROM python:3.12-slim

# ffmpeg converts reference clips at voice lock and backs librosa/audioread decoding.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# torch first and from the CPU index alone, so no later resolution can pull the
# CUDA wheel from PyPI; requirements.txt then finds it already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# resemblyzer's metadata demands the source-only `webrtcvad` (no compiler here) and
# the obsolete `typing` backport; its real deps are already installed above.
RUN pip install --no-cache-dir --no-deps resemblyzer==0.1.4

# Fail the build, not the first render: every audio dep imports, and the heavy deps
# the rebuild banned (M1 acceptance) are provably absent.
RUN python -c "import resemblyzer, webrtcvad, pedalboard, pyloudnorm, librosa" \
 && python -c "import importlib.util as u; assert u.find_spec('torchaudio') is None; assert u.find_spec('parselmouth') is None"

# Unbuffered stdout keeps `docker logs` live; librosa's numba JIT cache persists in
# the /cache volume so pyin does not recompile on every container start.
ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/cache/numba

WORKDIR /app
COPY configs/ /app/configs/
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY tests/golden/ /app/tests/golden/
COPY web/ /app/web/
# The shipped phrase bank; app.main seeds it into the /data volume on first boot
# so an operator can edit the live copy without a rebuild.
COPY data/phrases.json /app/data/phrases.json

EXPOSE 8020
# M2 service entrypoint. In M1 run one-shot commands instead:
#   docker compose -f docker/compose.yml run --rm app python -m scripts.bag_cli render --voice bag --text "..."
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]
