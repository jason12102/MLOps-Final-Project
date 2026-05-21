# ─────────────────────────────────────────────────────────────────────────────
# H2O AutoML Demo — MLOps Final Project
# Exposes:
#   :8888  → Jupyter Lab
#   :54321 → H2O Flow UI
# ─────────────────────────────────────────────────────────────────────────────

# Use a base image that already has Java bundled — avoids apt entirely
FROM eclipse-temurin:17-jre-jammy

# ── Install Python ────────────────────────────────────────────────────────────
RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' resolve without the '3' suffix
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Java is already in the temurin image — just confirm JAVA_HOME
ENV JAVA_HOME=/opt/java/openjdk

# ── Python packages ───────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --retries 5 -r /tmp/requirements.txt

# ── App directory ─────────────────────────────────────────────────────────────
WORKDIR /app
COPY notebooks/ /app/notebooks/

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 54321 8888

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
