# ✅ Use Python 3.10
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    python3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for layer caching)
COPY requirements.txt .

# ✅ Install dependencies including CPU Torch
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy all source code
COPY . .

# ✅ Expose Render runtime port
ENV PORT=8000
EXPOSE 8000

# ✅ Start FastAPI App
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
