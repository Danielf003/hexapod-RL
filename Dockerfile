FROM python:3.10-alpine
#3.10-slim
# Based on
# https://github.com/dporkka/docker-101?ysclid=m60umny7nw465329032
# https://hub.docker.com/_/python

# Install system dependencies
RUN apt-get update
RUN apt-get install -y git python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Install any python packages you need
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch CPU only
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

RUN git clone https://github.com/Danielf003/hexapod-RL

WORKDIR /app/hexapod-RL

# Set the entrypoint
CMD ["bash"]