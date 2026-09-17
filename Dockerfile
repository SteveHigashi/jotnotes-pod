# JotNotes Brilliant GPU pod: Ollama + qwen2.5:7b + Kokoro voice, ready at boot.
# Built by GitHub Actions; pulled by RunPod or Vast. The RunPod base keeps its
# own /start.sh (sshd, the PUBLIC_KEY it injects), so a pod stays reachable the
# way it always was; this image only adds the two services and starts them.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=8 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 ONNX_PROVIDER=CUDAExecutionProvider ARCA_TTS_PORT=9997
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates espeak-ng python3-venv && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# The model, pulled at build time so a pod never downloads it.
RUN (ollama serve >/tmp/ollama-build.log 2>&1 &) && for i in $(seq 1 60); do curl -s 127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done \
 && ollama pull qwen2.5:7b && pkill ollama || true

# The voice.
WORKDIR /opt/tts
COPY tts_server.py .
RUN python3 -m venv venv && ./venv/bin/pip install --quiet --upgrade pip \
 && ./venv/bin/pip install --quiet "kokoro-onnx==0.4.7" "onnxruntime-gpu==1.22.0" "soundfile==0.14.0" "numpy>=2,<3" \
 && curl -fL --retry 3 -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
 && curl -fL --retry 3 -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY pod-start.sh /pod-start.sh
RUN chmod +x /pod-start.sh
CMD ["/pod-start.sh"]
