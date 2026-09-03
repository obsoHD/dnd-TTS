# Light service: guarantees Bag's voice (F0 gate), shapes speed + space (ffmpeg),
# serves the browser UI. Talks to the heavy TTS container over the network.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" pydantic

WORKDIR /app
COPY server/ /app/server/

EXPOSE 8020
CMD ["uvicorn", "server.orchestrator:app", "--host", "0.0.0.0", "--port", "8020"]
