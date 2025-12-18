FROM pytorch/pytorch:2.2.2-cpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models /app/models
COPY run_infer.py /app/run_infer.py
COPY weights /app/weights

ENV PYTHONUNBUFFERED=1

# Default: expect input mounted at /input and output at /output
CMD ["python", "/app/run_infer.py", "--input", "/input", "--output", "/output", "--weights", "/app/weights/best_model.pth"]

