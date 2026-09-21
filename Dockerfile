# Pinned Python version so torch/transformers wheels are available regardless of the host.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights (~550MB) download on first run into /root/.cache; mount it as a volume to skip re-downloads (NOTES.md 1).
# Runs the local backend by default; for AWS override the command: docker run --network host <image> python run.py --backend aws
CMD ["python", "run.py", "--backend", "local", "--input", "data/conversations", "--work-dir", "queue", "--output", "results"]
