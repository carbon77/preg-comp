FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# App directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock* requirements.txt* ./

RUN uv sync --locked

# Copy project
COPY . .

# Streamlit port
EXPOSE 8501

# Streamlit settings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run app
CMD ["uv", "run", "streamlit", "run", "main.py"]