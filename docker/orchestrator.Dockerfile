# Bag orchestrator: guarantees the voice (F0 gate), shapes speed + space (ffmpeg),
# and does deterministic vowel elongation (MMS forced alignment, CPU). Serves the
# browser UI. Talks to the heavy TTS container over the network.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" pydantic \
      requests python-multipart numpy pedalboard pyloudnorm \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio

# MMS_FA alignment model caches here; mount a volume to persist across restarts.
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/hf

WORKDIR /app
COPY server/ /app/server/

EXPOSE 8020
CMD ["uvicorn", "server.orchestrator:app", "--host", "0.0.0.0", "--port", "8020"]
