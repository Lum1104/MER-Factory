# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    git \
    wget \
    unzip \
    bzip2 \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    # FFmpeg
    ffmpeg \
    # Compilers
    g++-11 \
    gcc-11 \
    # OpenBLAS
    libopenblas-dev \
    # GTK and GUI libraries
    libgtk2.0-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    # OpenCV dependencies
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    libtbb2 \
    libtbb-dev \
    # X virtual framebuffer (for headless operation)
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install OpenCV 4.12.0
RUN wget -q https://github.com/opencv/opencv/archive/4.12.0.zip -O /tmp/opencv.zip && \
    cd /tmp && \
    unzip opencv.zip && \
    cd opencv-4.12.0 && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TIFF=ON \
          -D WITH_TBB=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/opencv*

# Install dlib 19.13
RUN wget -q http://dlib.net/files/dlib-19.13.tar.bz2 -O /tmp/dlib.tar.bz2 && \
    cd /tmp && \
    tar xf dlib.tar.bz2 && \
    cd dlib-19.13 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/dlib*

# Fix compatibility issue between new OpenCV and old dlib
RUN sed -i 's/IplImage temp = img;/IplImage temp = cvIplImage(img);/g' /usr/local/include/dlib/opencv/cv_image.h

# Clone and build OpenFace
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /app/OpenFace && \
    cd /app/OpenFace && \
    mkdir -p build && \
    cd build && \
    cmake -D CMAKE_CXX_COMPILER=g++-11 \
          -D CMAKE_C_COMPILER=gcc-11 \
          -D CMAKE_BUILD_TYPE=RELEASE .. && \
    make -j$(nproc)

# Download OpenFace models
RUN cd /app/OpenFace/lib/local/LandmarkDetector/model/patch_experts && \
    wget -q https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat || \
    wget -q https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153072&authkey=AKqoZtcN0PSIZH4 -O cen_patches_0.25_of.dat && \
    wget -q https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat || \
    wget -q https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153079&authkey=ANpDR1n3ckL_0gs -O cen_patches_0.35_of.dat && \
    wget -q https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat || \
    wget -q https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153074&authkey=AGi-e30AfRc_zvs -O cen_patches_0.50_of.dat && \
    wget -q https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat || \
    wget -q https://onedrive.live.com/download?cid=2E2ADA578BFF6E6E&resid=2E2ADA578BFF6E6E%2153070&authkey=AD6KjtYipphwBPc -O cen_patches_1.00_of.dat

# Copy patch experts to OpenFace build directory
RUN mkdir -p /app/OpenFace/build/bin/model && \
    cp -r /app/OpenFace/lib/local/LandmarkDetector/model/patch_experts /app/OpenFace/build/bin/model/

# Copy MER-Factory project files
COPY . /app/MER-Factory

# Set working directory to MER-Factory
WORKDIR /app/MER-Factory

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create .env file with default values
RUN echo "# Google Gemini API Key (get from https://makersuite.google.com/app/apikey)" > .env && \
    echo "GOOGLE_API_KEY=your_google_gemini_api_key_here" >> .env && \
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env && \
    echo "" >> .env && \
    echo "# OpenFace FeatureExtraction executable path" >> .env && \
    echo "OPENFACE_EXECUTABLE=/app/OpenFace/build/bin/FeatureExtraction" >> .env && \
    echo "" >> .env && \
    echo "# Url to your hf server" >> .env && \
    echo "HF_API_BASE_URL=\"http://localhost:7860/\"" >> .env

# Modify openface_adapter.py to use xvfb-run for headless operation
RUN sed -i 's/openface_executable,/"xvfb-run", openface_executable,/g' tools/openface_adapter.py

# Create output directory
RUN mkdir -p /app/MER-Factory/output

# Expose port for gradio dashboard (if needed)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENFACE_EXECUTABLE=/app/OpenFace/build/bin/FeatureExtraction

# Default command - can be overridden
CMD ["python", "main.py", "--help"]

