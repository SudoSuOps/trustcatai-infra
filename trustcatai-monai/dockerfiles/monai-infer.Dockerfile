FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "uvicorn[standard]"

COPY trustcatai /workspace/trustcatai
COPY pyproject.toml /workspace/

ENV PYTHONPATH=/workspace

EXPOSE 8080

ENTRYPOINT ["uvicorn", "trustcatai.infer.server:app", "--host", "0.0.0.0", "--port", "8080"]
