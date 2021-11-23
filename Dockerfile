ARG TRT_IMAGE_VERSION=20.09
FROM nvcr.io/nvidia/tensorrt:${TRT_IMAGE_VERSION}-py3

ARG TRT_IMAGE_VERSION
ARG OPENCV_VERSION=4.1.1
ARG APP_DIR=/usr/src/app
ARG SCRIPT_DIR=/opt/tensorrt/python
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=${APP_DIR}
ENV TZ=America/Los_Angeles

ENV OPENBLAS_MAIN_FREE=1
ENV OPENBLAS_NUM_THREADS=1
ENV NO_AT_BRIDGE=1

# Install OpenCV and FastMOT dependencies
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    wget unzip tzdata \
    build-essential cmake pkg-config \
    libgtk-3-dev libcanberra-gtk3-module \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    gfortran libatlas-base-dev \
    python3-dev \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libtbb2 libtbb-dev libdc1394-22-dev \
    ffmpeg && \
    pip install -U --no-cache-dir setuptools pip && \
    pip install --no-cache-dir numpy==1.18.0

# Build OpenCV
WORKDIR ${HOME}
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv_contrib-${OPENCV_VERSION} OpenCV/opencv_contrib

# If you have issues with GStreamer, set -DWITH_GSTREAMER=OFF and -DWITH_FFMPEG=ON
WORKDIR ${HOME}/OpenCV/build
RUN cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=${HOME}/OpenCV/opencv_contrib/modules \
    -DINSTALL_PYTHON_EXAMPLES=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_PROTOBUF=OFF \
    -DENABLE_FAST_MATH=ON \
    -DWITH_TBB=ON \
    -DWITH_LIBV4L=ON \
    -DWITH_CUDA=OFF \
    -DWITH_GSTREAMER=ON \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_FFMPEG=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf ${HOME}/OpenCV && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove

# Install Python dependencies
WORKDIR ${APP_DIR}/FastMOT
COPY requirements.txt .

# Specify your GPU compute with --build-arg for CuPy (e.g. "arch=compute_75,code=sm_75")
ARG CUPY_NVCC_GENERATE_CODE

# TensorFlow < 2 is not supported in ubuntu 20.04
RUN if [[ -z ${CUPY_NVCC_GENERATE_CODE} ]]; then \
        echo "CUPY_NVCC_GENERATE_CODE not set, building CuPy for all architectures (slower)"; \
    fi && \
    if dpkg --compare-versions ${TRT_IMAGE_VERSION} ge 20.12; then \
        CUPY_NUM_BUILD_JOBS=$(nproc) pip install --no-cache-dir -r <(grep -ivE "tensorflow" requirements.txt); \
    else \
        dpkg -i ${SCRIPT_DIR}/*-tf_*.deb && \
        CUPY_NUM_BUILD_JOBS=$(nproc) pip install --no-cache-dir -r requirements.txt; \
    fi

RUN apt update && \
    git config --global http.sslverify false && \
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers &&  make install && cd -- && \
    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ && \
    apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev && \
    cd ffmpeg && \
    ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared && \
    make -j $(nproc) && \
    make install

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"


RUN cd ~ && \
    git clone https://github.com/FFmpeg/FFmpeg.git && \
    cd FFmpeg && \
    git checkout tags/n4.4 && \
    mkdir -p $(pwd)/build_x64_release_shared && \
    ./configure \
    --prefix=$(pwd)/build_x64_release_shared \
    --disable-static \
    --disable-stripping \
    --disable-doc \
    --enable-shared && \
    make -j -s && make install

RUN pip install torch torchvision torchaudio

ENV PATH_TO_SDK=/usr/src/app/Video_Codec_SDK_11.1.5
ENV PATH_TO_FFMPEG=/usr/src/app/FFmpeg/build_x64_release_shared
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV INSTALL_PREFIX=/usr/src/app/VideoProcessingFramework/install

COPY Video_Codec_SDK_11.1.5 /usr/src/app/Video_Codec_SDK_11.1.5

RUN cd ~ && \
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git && \
    cd VideoProcessingFramework && \
    mkdir -p install && \
    mkdir -p build && \
    cd build && \
    cmake .. -DFFMPEG_DIR:PATH=/usr/src/app/FFmpeg/build_x64_release_shared \
    -DVIDEO_CODEC_SDK_DIR:PATH=/usr/src/app/Video_Codec_SDK_11.1.5 \
    -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
    -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
    -DCMAKE_INSTALL_PREFIX:PATH=/usr/src/app/VideoProcessingFramework/install \
    -DAVCODEC_INCLUDE_DIR:PATH="/usr/src/app/FFmpeg/build_x64_release_shared/include" \
    -DAVFORMAT_INCLUDE_DIR:PATH="/usr/src/app/FFmpeg/build_x64_release_shared/include" \
    -DAVUTIL_INCLUDE_DIR:PATH="/usr/src/app/FFmpeg/build_x64_release_shared/include" \
    -DSWRESAMPLE_LIBRARY="/usr/src/app/FFmpeg/build_x64_release_shared/lib/libswresample.so" \
    -DAVFORMAT_LIBRARY="/usr/src/app/FFmpeg/build_x64_release_shared/lib/libavformat.so" \
    -DAVCODEC_LIBRARY="/usr/src/app/FFmpeg/build_x64_release_shared/lib/libavcodec.so" \
    -DAVUTIL_LIBRARY="/usr/src/app/FFmpeg/build_x64_release_shared/lib/libavutil.so" && \
    make && make install && \
    ldd $INSTALL_PREFIX/bin/PyNvCodec.cpython-38-x86_64-linux-gnu.so

ENV LD_LIBRARY_PATH=$PATH_TO_FFMPEG/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$INSTALL_PREFIX/bin:$LD_LIBRARY_PATH

# TODO: workarround, find proper solution
RUN cp /usr/src/app/VideoProcessingFramework/install/bin/PyNvCodec.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages


# ------------------------------------  Extras Below  ------------------------------------

# Stop the container (changes are kept)
# docker stop $(docker ps -ql)

# Start the container
# docker start -ai $(docker ps -ql)

# Delete the container
# docker rm $(docker ps -ql)

# Save changes before deleting the container
# docker commit $(docker ps -ql) fastmot:latest