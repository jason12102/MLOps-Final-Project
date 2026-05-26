FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/workspace /app/data

COPY config.yaml /app/config.yaml
COPY src/ /app/

EXPOSE 8001

CMD ["python", "main.py", "--mode", "wait-and-run"]
