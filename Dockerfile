<<<<<<< HEAD
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app2.py", "--server.port=8501", "--server.enableCORS=false"]
=======
# Use a Debian-based image with Python
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Clone llama-cpp-python (if used in editable mode)
# You can skip this if you install via pip in requirements.txt
# RUN git clone https://github.com/abetlen/llama-cpp-python.git && \
#     cd llama-cpp-python && \
#     pip install .

# Copy all source code
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
>>>>>>> ecccdfb6f44d19ebdd68fa56f0faa987c67ebecc
