#!/usr/bin/env bash
# install_openpose_colab.sh -- build CMU OpenPose on a Colab GPU runtime.
#
# Exercised on Colab (Ubuntu 24.04 image, CUDA 12, cuDNN 9.8, protobuf
# 3.21) as far as the Caffe build; each break found so far is handled below.
# OpenPose is a C++/CUDA build with a 2017-era Caffe inside, so expect that
# a new image can surface a new one -- the script prints the last lines of
# the failing log when it does.
# Expect 20-40 minutes.  Run once per session; the result does not survive
# the runtime, so consider tarring build/ + models/ to Drive afterwards.
#
#     !bash datasets/biocv/install_openpose_colab.sh /content/openpose [tarball]
# then
#     python3 datasets/biocv/run_openpose.py \
#         --openpose-bin /content/openpose/build/examples/openpose/openpose.bin \
#         --model-folder /content/openpose/models/
#
# Restore vs build.  If the binary already exists, nothing is built.  Else if
# the tarball exists (default /content/drive/MyDrive/openpose_built.tgz -- mount
# Drive first), it is unpacked and the build is skipped (~1 min).  Else the
# full build runs and, if Drive is mounted, the tarball is written at the end
# so the next session restores.  The tarball must unpack to the SAME path it
# was built at: the binary finds libopenpose/libcaffe through absolute RPATHs
# into the build tree.  The apt libraries are runtime dependencies and are
# reinstalled every session either way (fast).
#
# Needs a GPU runtime (Runtime -> Change runtime type -> T4 GPU): on a CPU
# runtime there are no CUDA libraries and the binary fails with rc=127.  If the
# image's CUDA is newer than the one the tarball was built on, the missing
# runtime libraries are fetched as pip wheels (see cuda_libs below).
set -euo pipefail

ROOT=${1:-/content/openpose}
TAR=${2:-/content/drive/MyDrive/openpose_built.tgz}
JOBS=$(nproc)
BIN="$ROOT/build/examples/openpose/openpose.bin"
MODEL="$ROOT/models/pose/body_25/pose_iter_584000.caffemodel"

# CUDA runtime libraries the binary was linked against but this image lacks.
# The tarball was built on a CUDA 12 image (libcudart.so.12, libcublas.so.12);
# by Nov 2026 Colab's GPU image ships CUDA 13 (.so.13 -- libcurand keeps .so.10
# across both, so it still resolves).  The driver runs CUDA 12 binaries, so the
# CUDA 12 runtime libraries are installed as pip wheels and registered with the
# dynamic loader (ld.so.conf.d + ldconfig) -- unlike LD_LIBRARY_PATH, that
# holds for every later notebook cell.  A no-op when nothing is missing.
cuda_libs() {
    local missing
    missing=$(ldd "$BIN" | awk '/not found/ {print $1}' | grep -E '^lib(cudart|cublas|cublasLt|curand)\.so\.' || true)
    [ -z "$missing" ] && return 0
    echo "### CUDA runtime libraries missing: $(echo $missing) ###"
    local major pkgs=""
    major=$(echo "$missing" | head -1 | sed 's/.*\.so\.//')
    case "$missing" in *cudart*) pkgs="$pkgs nvidia-cuda-runtime-cu$major" ;; esac
    case "$missing" in *cublas*) pkgs="$pkgs nvidia-cublas-cu$major" ;; esac
    if echo "$missing" | grep -q curand; then
        echo "libcurand missing too -- cannot tell its CUDA version from the .so name; trying cu$major"
        pkgs="$pkgs nvidia-curand-cu$major"
    fi
    python3 -m pip -q install $pkgs
    # register only the folders holding the missing libraries, so nothing else
    # on the image (e.g. torch's own CUDA wheels) changes which libraries it loads
    python3 - $missing > /etc/ld.so.conf.d/openpose-cuda.conf <<'PY'
import glob, os, site, sys
for d in sorted({os.path.dirname(p) for sp in site.getsitepackages() for name in sys.argv[1:]
                 for p in glob.glob(f'{sp}/nvidia/*/lib/{name}')}):
    print(d)
PY
    ldconfig
    echo "registered: $(tr '\n' ' ' < /etc/ld.so.conf.d/openpose-cuda.conf)"
    missing=$(ldd "$BIN" | awk '/not found/ {print $1}' || true)
    if [ -n "$missing" ]; then
        echo "still missing: $(echo $missing)"; exit 1
    fi
}

smoke() {
    cuda_libs
    echo; echo "binary : $BIN"; echo "model  : $MODEL ($(stat -c %s "$MODEL") bytes)"
    echo "smoke test:"; "$BIN" --help 2>&1 | head -3
}

echo "### apt dependencies ###"
apt-get -qq update
apt-get -qq install -y cmake libopencv-dev protobuf-compiler libprotobuf-dev \
    libgoogle-glog-dev libgflags-dev libboost-all-dev libhdf5-dev libatlas-base-dev \
    libleveldb-dev libsnappy-dev liblmdb-dev > /dev/null

if [ -x "$BIN" ] && [ -s "$MODEL" ]; then
    echo "### already built at $ROOT ###"; smoke; exit 0
fi
if [ -f "$TAR" ]; then
    echo "### restoring from $TAR ###"
    mkdir -p "$(dirname "$ROOT")"
    tar xzf "$TAR" -C "$(dirname "$ROOT")"
    if [ -x "$BIN" ] && [ -s "$MODEL" ]; then smoke; exit 0; fi
    echo "tarball did not contain $BIN and $MODEL -- building from source"
fi

echo "### source ###"
if [ ! -d "$ROOT" ]; then
    git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git "$ROOT"
fi
cd "$ROOT"
git submodule update --init --recursive --remote -q

echo "### configure ###"
# USE_CUDNN=OFF: OpenPose's bundled Caffe predates cuDNN 8 and fails to
# compile against it on current images.  Slower inference, but it builds.
# CUDA_ARCH: T4 = 75, A100 = 80, L4 = 89.  Manual so cmake does not probe.
mkdir -p build && cd build
cmake .. \
    -DBUILD_PYTHON=OFF -DBUILD_EXAMPLES=ON \
    -DUSE_CUDNN=OFF \
    -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN="75 80 86 89" -DCUDA_ARCH_PTX="" \
    -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_BODY_COCO_MODEL=OFF \
    -DDOWNLOAD_BODY_MPI_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF \
    > cmake.log 2>&1 || { tail -30 cmake.log; exit 1; }

echo "### BODY_25 weights ###"
# The CMU model server cmake would download from is dead (HTTP error), and
# with the download left ON the configure step fails outright -- so it is OFF
# above and the weights come from a Hugging Face mirror of the OpenPose repo
# instead.  104,715,850 bytes is the genuine file; the ~200 MB one is COCO.
MIRROR="https://huggingface.co/camenduru/openpose/resolve/main/models/pose/body_25/pose_iter_584000.caffemodel"
if [ ! -s "$MODEL" ] || [ "$(stat -c %s "$MODEL")" -lt 100000000 ]; then
    mkdir -p "$(dirname "$MODEL")"
    wget -q --show-progress -O "$MODEL" "$MIRROR" || { echo "model download failed"; exit 1; }
fi
echo "model: $(stat -c %s "$MODEL") bytes"

echo "### patch Caffe for current protobuf ###"
# Caffe calls the two-argument SetTotalBytesLimit(limit, warning) that
# protobuf removed in 3.x; the one-argument form is what every image now
# ships.  Applied after cmake, since cmake is what checks Caffe out.
# Idempotent: sed matches nothing on a second run.
IO="$ROOT/3rdparty/caffe/src/caffe/util/io.cpp"
sed -i 's/SetTotalBytesLimit(kProtoReadBytesLimit, 536870912)/SetTotalBytesLimit(kProtoReadBytesLimit)/' "$IO"
grep -n "SetTotalBytesLimit" "$IO"
if grep -q "SetTotalBytesLimit(kProtoReadBytesLimit, 536870912)" "$IO"; then
    echo "protobuf patch did not apply to $IO"; exit 1
fi

echo "### build (this is the slow part) ###"
make -j"$JOBS" > make.log 2>&1 || { tail -40 make.log; exit 1; }

[ -x "$BIN" ] || { echo "no binary built"; exit 1; }
smoke

# Save for next session: the binary, the two shared libraries it links, and
# the models.  Skipped if Drive is not mounted.
if [ -d "$(dirname "$TAR")" ]; then
    echo; echo "### saving $TAR ###"
    R=$(basename "$ROOT")
    tar czf "$TAR" -C "$(dirname "$ROOT")" \
        "$R/build/examples/openpose/openpose.bin" "$R/build/src" "$R/build/caffe/lib" "$R/models"
    echo "saved $(stat -c %s "$TAR") bytes"
else
    echo; echo "(Drive not mounted at $(dirname "$TAR") -- not saving a tarball)"
fi
