# JotNotes Brilliant GPU pod image

Ollama with qwen2.5:7b and the Kokoro voice, ready at boot, on RunPod's PyTorch base so SSH works as
RunPod expects. Image: `ghcr.io/stevehigashi/jotnotes-pod:latest`. Ports inside: 11434 (engine),
9997 (voice), both on the loopback; reach them over an SSH tunnel, never exposed.
