import os, ctypes.util
import cupy as cp
from cupy.cuda import runtime
import cupy_backends.cuda.libs.nvrtc as nvrtc

print("CuPy:", cp.__version__)
print("CUDA runtime:", runtime.runtimeGetVersion())
print("Driver:", runtime.driverGetVersion())
print("NVRTC lib:", ctypes.util.find_library("nvrtc"))
print("CUDA include:", os.environ.get("CUPY_NVRTC_INC_PATH"))
x = cp.arange(8, dtype=cp.float32); y = x**2
print("OK:", y.get())
