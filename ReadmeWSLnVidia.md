## Instructions to detect and use nVidia GPUs on WSL
### Ubuntu cleanup needed to fix 404s on the security repo
```bash
sudo apt clean
sudo dpkg --configure -a
sudo apt update
sudo apt --fix-broken install
```
```bash
mkdir ~/Downloads
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo apt install -y gcc # required by next cmd
sudo bash ./cuda_12.8.1_570.124.06_linux.run --silent --toolkit --no-opengl-libs
```
```bash
sudo apt install -y python3-dev libpython3-dev build-essential ocl-icd-libopencl1 cmake git pkg-config make ninja-build ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev clinfo dialog apt-utils libxml2-dev opencl-headers
```
```bash
export LLVM_VERSION=14
sudo apt install -y libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} llvm-${LLVM_VERSION}  libclang-cpp${LLVM_VERSION}-dev libclang-cpp${LLVM_VERSION} llvm-${LLVM_VERSION}-dev 
```
```bash
git clone https://github.com/pocl/pocl -b v6.0
mkdir pocl/build
cd pocl/build
cmake -DCMAKE_C_FLAGS=-L/usr/lib/wsl/lib \
  -DCMAKE_CXX_FLAGS=-L/usr/lib/wsl/lib \
  -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-14 \
  -DENABLE_HOST_CPU_DEVICES=OFF \
  -DENABLE_CUDA=ON ..
```
```bash
make -j`nproc`
sudo make install
sudo mkdir -p /etc/OpenCL/vendors # path missing on default install
sudo cp /usr/local/etc/OpenCL/vendors/pocl.icd /etc/OpenCL/vendors/pocl.icd # We need this otherwise `clinfo` returns 0 platform detected
```