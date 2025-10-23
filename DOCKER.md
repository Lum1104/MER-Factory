# MER-Factory Docker Guide

This guide provides instructions for building and running MER-Factory in a Docker container.

## Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- At least 8GB of disk space for the image
- (Optional) GPU support for faster processing

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t mer-factory:latest .
```

This will take approximately 15-30 minutes as it builds OpenFace and OpenCV from source.

### 2. Configure API Keys

Before running the container, you'll need to set up your API keys. You have two options:

#### Option A: Use Environment Variables (Recommended)

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_actual_google_api_key \
  -e OPENAI_API_KEY=your_actual_openai_api_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/your_video.mp4 /app/MER-Factory/output --type MER --silent
```

#### Option B: Mount a Custom .env File

Create a `.env` file locally with your keys:

```bash
# .env
GOOGLE_API_KEY=your_actual_google_api_key
OPENAI_API_KEY=your_actual_openai_api_key
OPENFACE_EXECUTABLE=/app/OpenFace/build/bin/FeatureExtraction
HF_API_BASE_URL="http://localhost:7860/"
```

Then mount it when running:

```bash
docker run -it --rm \
  -v $(pwd)/.env:/app/MER-Factory/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/your_video.mp4 /app/MER-Factory/output --type MER --silent
```

## Usage Examples

### Process a Single Video

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

### Process Multiple Videos in a Directory

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/videos /app/MER-Factory/output --type MER --silent
```

### Process Images

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/images:/app/images \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/images /app/MER-Factory/output --type image
```

### Run with ChatGPT Models

```bash
docker run -it --rm \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER --chatgpt-model gpt-4o
```

### Action Unit (AU) Analysis Only

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type AU
```

### Audio Analysis Only

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type audio
```

### Run Dashboard

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/output:/app/MER-Factory/output \
  -p 7860:7860 \
  mer-factory:latest \
  python dashboard.py
```

Then access the dashboard at `http://localhost:7860`

### Interactive Shell

To get an interactive shell inside the container:

```bash
docker run -it --rm \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  /bin/bash
```

## Windows Usage

If you're on Windows, replace `$(pwd)` with `%cd%` for Command Prompt or `${PWD}` for PowerShell:

### Command Prompt:
```cmd
docker run -it --rm ^
  -e GOOGLE_API_KEY=your_key ^
  -v %cd%/data:/app/data ^
  -v %cd%/output:/app/MER-Factory/output ^
  mer-factory:latest ^
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

### PowerShell:
```powershell
docker run -it --rm `
  -e GOOGLE_API_KEY=your_key `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/output:/app/MER-Factory/output `
  mer-factory:latest `
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

## GPU Support (Optional)

To use GPU acceleration with Docker, you need NVIDIA Docker runtime:

```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU
docker run -it --rm --gpus all \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

## Volume Mounts Explained

- `-v $(pwd)/data:/app/data` - Mount your local data directory containing videos/images
- `-v $(pwd)/output:/app/MER-Factory/output` - Mount output directory to persist results
- `-v $(pwd)/.env:/app/MER-Factory/.env` - Mount environment configuration file

## Troubleshooting

### Permission Issues
If you encounter permission issues with output files:

```bash
# Run with user ID mapping
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

### OpenFace Model Download Failed
If OpenFace models fail to download during build, try rebuilding with:

```bash
docker build --no-cache -t mer-factory:latest .
```

### Out of Memory
If processing fails due to memory issues, increase Docker's memory limit in Docker Desktop settings or add `--memory` flag:

```bash
docker run -it --rm --memory=8g \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/MER-Factory/output \
  mer-factory:latest \
  python main.py /app/data/video.mp4 /app/MER-Factory/output --type MER
```

## Docker Compose (Alternative)

For easier management, you can create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mer-factory:
    build: .
    image: mer-factory:latest
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./output:/app/MER-Factory/output
    command: python main.py /app/data /app/MER-Factory/output --type MER --silent
```

Then run with:

```bash
# Set your API key
export GOOGLE_API_KEY=your_actual_key

# Run
docker-compose up
```

## Cleanup

To remove the Docker image:

```bash
docker rmi mer-factory:latest
```

To clean up all unused Docker resources:

```bash
docker system prune -a
```

## Notes

- The Docker image includes all dependencies (FFmpeg, OpenFace, OpenCV, dlib)
- OpenFace runs in headless mode using xvfb
- All processing is done inside the container
- Results are persisted to your local machine via volume mounts
- The image size is approximately 4-5GB due to compilation of OpenCV and OpenFace

## Support

For issues specific to Docker setup, please check:
- [Docker Documentation](https://docs.docker.com/)
- [MER-Factory Issues](https://github.com/Lum1104/MER-Factory/issues)

