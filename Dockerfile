# Pinned to a known-good Python version for torch/transformers wheel
# availability, independent of whatever Python the host machine has.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights (~550MB) are downloaded on first run into /root/.cache -
# mount that as a volume in production so a restarted container does not
# re-download them. Nothing else is written outside /app/queue and
# /app/results, both inside the container filesystem. Verified: a second
# `docker run` against the same cache volume took ~7s instead of ~40s and
# made 6 HTTP calls instead of 25+, all lightweight revision checks with no
# weight re-download - see NOTES.md, Section 1.
#
# Runs the local, zero-infra backend by default. To run against SQS/S3/
# DynamoDB instead (see docker-compose.yml), override the command - verified
# working on Docker Desktop for Windows (WSL2 backend), see NOTES.md Section 6:
#   docker run --network host <image> python run.py --backend aws
CMD ["python", "run.py", "--backend", "local", "--input", "data/conversations", "--work-dir", "queue", "--output", "results"]
