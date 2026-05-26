FROM eclipse-temurin:17-jre-jammy

RUN apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

ENV JAVA_HOME=/opt/java/openjdk

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --retries 5 -r requirements.txt

COPY app/ /app/
COPY model/ /app/model/

ENV MODEL_PATH=/app/model/GBM_1_AutoML_3_20260525_194002

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
