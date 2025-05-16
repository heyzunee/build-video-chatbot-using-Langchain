FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /main
COPY requirements.txt .

RUN apt-get -y update --no-install-recommends && \
    apt-get -y install --no-install-recommends \
    git gcc g++ cmake curl build-essential \
    python3-dev nano vim openssh-client \
    pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--browser.serverAddress=192.168.11.11"]
ENTRYPOINT ["python", "app.py"]
