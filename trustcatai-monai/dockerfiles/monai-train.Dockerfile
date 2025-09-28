FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TRUSTCAT_HOME=/workspace

WORKDIR ${TRUSTCAT_HOME}

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY trustcatai ./trustcatai
COPY pyproject.toml ./pyproject.toml

ENV PYTHONPATH=${TRUSTCAT_HOME}

ENTRYPOINT ["python", "-m", "trustcatai.trainer.agent"]
