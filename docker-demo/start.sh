#!/bin/bash
set -e

echo "────────────────────────────────────────────"
echo " H2O AutoML Demo — MLOps Final Project"
echo " H2O Flow  → http://localhost:54321"
echo " Jupyter   → http://localhost:8888"
echo "────────────────────────────────────────────"

# ── Find h2o.jar ──────────────────────────────────────────────────────────────
H2O_JAR=$(python -c "import h2o, os; print(os.path.join(os.path.dirname(h2o.__file__), 'backend', 'bin', 'h2o.jar'))")
echo "H2O jar: $H2O_JAR"

if [ ! -f "$H2O_JAR" ]; then
  echo "ERROR: h2o.jar not found at $H2O_JAR"
  exit 1
fi

# ── Start H2O in background ───────────────────────────────────────────────────
java -Xmx4g \
  -jar "$H2O_JAR" \
  -nthreads -1 \
  -port 54321 \
  -ice_root /tmp/h2o_ice \
  -name demo_cluster \
  > /tmp/h2o.log 2>&1 &
H2O_PID=$!
echo "H2O started (PID $H2O_PID), waiting for port 54321..."

# ── Wait with timeout ─────────────────────────────────────────────────────────
TIMEOUT=60
ELAPSED=0
until curl -sf http://localhost:54321/3/Cloud > /dev/null 2>&1; do
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: H2O did not start within ${TIMEOUT}s. Log:"
    tail -30 /tmp/h2o.log
    exit 1
  fi
  if ! kill -0 $H2O_PID 2>/dev/null; then
    echo "ERROR: H2O process died. Log:"
    tail -30 /tmp/h2o.log
    exit 1
  fi
done
echo "H2O is up."

# ── Start Jupyter Lab ─────────────────────────────────────────────────────────
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --notebook-dir=/app/notebooks
