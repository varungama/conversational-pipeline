# Pinned to a known-good Python version for torch/transformers wheel
# availability, independent of whatever Python the host machine has.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights (~256MB) are downloaded on first run into /root/.cache -
# mount that as a volume in production so a restarted container does not
# re-download them. Nothing else is written outside /app/queue and
# /app/results, both inside the container filesystem.
CMD ["python", "run.py", "--input", "data/conversations", "--work-dir", "queue", "--output", "results"]
