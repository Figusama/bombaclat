# Вариант для внутреннего контура на Linux + NVIDIA GPU (бэкенд transformers, VLM_BACKEND=hf).
# На Apple Silicon сервис запускается нативно (run.sh), т.к. MLX внутри Docker недоступен.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements-linux.txt .
RUN pip install --no-cache-dir -r requirements-linux.txt
COPY app app
COPY pipeline pipeline
COPY catalog catalog
COPY tools tools
ENV VLM_BACKEND=hf MODEL_ID=Qwen/Qwen3.5-9B HF_HUB_OFFLINE=1 PORT=8000
# веса кладутся в /root/.cache/huggingface (volume) заранее: hf download Qwen/Qwen3.5-9B; hf download PekingU/rtdetr_r50vd
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
