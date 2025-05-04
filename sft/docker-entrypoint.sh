#!/bin/bash
set -e

# Print GPU information
echo "=== GPU Information ==="
if nvidia-smi > /dev/null 2>&1; then
    nvidia-smi
    echo "GPU detected and working properly."
else
    echo "Warning: GPU not detected. Running on CPU."
fi

# Set ulimit to allow core dumps
ulimit -c unlimited

# Monitor memory usage in background
(while true; do echo "=== $(date): Memory Status ==="; free -h; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; sleep 10; done) > memory_log.txt 2>&1 &
MONITOR_PID=$!

# Run the application with error redirection
echo "=== Starting CouchGPT ==="
pipenv run uvicorn sft:app --host 0.0.0.0 --port 9000
echo "API is now running at http://localhost:9000/sft/inference"


# Check if application crashed
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "=== Application crashed with exit code $EXIT_CODE ==="
  echo "=== Last 20 lines of error log: ==="
  tail -n 20 error.log
  
  # Check CUDA memory issues
  echo "=== Final GPU state ==="
  nvidia-smi
fi

# Stop the memory monitoring
kill $MONITOR_PID
exit $EXIT_CODE