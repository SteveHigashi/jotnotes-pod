#!/bin/bash
# Start the engine and the voice, then hand over to the base image's own
# start script (sshd, jupyter if asked for) so the pod behaves as RunPod expects.
mkdir -p /root/.ollama
(setsid nohup ollama serve >/var/log/ollama.log 2>&1 </dev/null &)
(cd /opt/tts && setsid nohup ./venv/bin/python tts_server.py >/var/log/tts.log 2>&1 </dev/null &)
# Warm the model so the first question is fast.
(for i in $(seq 1 60); do curl -s 127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done; curl -s 127.0.0.1:11434/api/generate -d '{"model":"qwen2.5:7b","prompt":"hi","stream":false}' >/dev/null 2>&1 &)
exec /start.sh
