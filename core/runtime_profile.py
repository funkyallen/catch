import os
from contextlib import nullcontext
from typing import Any, Dict

import torch


LOGICAL_CPU_COUNT = max(1, os.cpu_count() or 1)
PHYSICAL_CORE_ESTIMATE = max(1, LOGICAL_CPU_COUNT // 2) if LOGICAL_CPU_COUNT >= 16 else LOGICAL_CPU_COUNT

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    _GPU_PROPS = torch.cuda.get_device_properties(0)
    GPU_NAME = _GPU_PROPS.name
    GPU_MEMORY_GB = round(_GPU_PROPS.total_memory / (1024 ** 3), 2)
else:
    GPU_NAME = None
    GPU_MEMORY_GB = 0.0

# XGBoost usually saturates before using all logical CPUs. Leaving a few cores free
# prevents contention with preprocessing and logging on large multi-core servers.
RECOMMENDED_XGB_N_JOBS = max(4, min(PHYSICAL_CORE_ESTIMATE, 16))
RECOMMENDED_JOBLIB_CPUS = RECOMMENDED_XGB_N_JOBS
RECOMMENDED_TORCH_THREADS = max(4, min(PHYSICAL_CORE_ESTIMATE, 16))
RECOMMENDED_DATALOADER_WORKERS = 0 if CUDA_AVAILABLE else max(2, min(PHYSICAL_CORE_ESTIMATE, 8))
RECOMMENDED_INFERENCE_BATCH_SIZE = 4096 if GPU_MEMORY_GB >= 20 else 2048 if GPU_MEMORY_GB >= 10 else 1024
RECOMMENDED_XGB_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
PIN_MEMORY = CUDA_AVAILABLE
AMP_ENABLED = CUDA_AVAILABLE


def configure_runtime_profile() -> Dict[str, Any]:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(RECOMMENDED_JOBLIB_CPUS)
    os.environ["OMP_NUM_THREADS"] = str(RECOMMENDED_XGB_N_JOBS)
    os.environ["MKL_NUM_THREADS"] = str(RECOMMENDED_XGB_N_JOBS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(RECOMMENDED_XGB_N_JOBS)
    try:
        torch.set_num_threads(RECOMMENDED_TORCH_THREADS)
        torch.set_num_interop_threads(max(1, min(4, RECOMMENDED_TORCH_THREADS // 2)))
    except RuntimeError:
        pass

    profile = {
        "logical_cpu_count": LOGICAL_CPU_COUNT,
        "physical_core_estimate": PHYSICAL_CORE_ESTIMATE,
        "recommended_xgb_n_jobs": RECOMMENDED_XGB_N_JOBS,
        "recommended_joblib_cpus": RECOMMENDED_JOBLIB_CPUS,
        "recommended_torch_threads": RECOMMENDED_TORCH_THREADS,
        "recommended_dataloader_workers": RECOMMENDED_DATALOADER_WORKERS,
        "recommended_inference_batch_size": RECOMMENDED_INFERENCE_BATCH_SIZE,
        "recommended_xgb_device": RECOMMENDED_XGB_DEVICE,
        "pin_memory": PIN_MEMORY,
        "amp_enabled": AMP_ENABLED,
        "cuda_available": CUDA_AVAILABLE,
    }

    if CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        profile.update(
            {
                "gpu_name": GPU_NAME,
                "gpu_memory_gb": GPU_MEMORY_GB,
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            }
        )
    else:
        profile.update(
            {
                "gpu_name": None,
                "gpu_memory_gb": 0.0,
                "tf32_matmul": False,
                "tf32_cudnn": False,
                "cudnn_benchmark": False,
            }
        )

    return profile


def autocast_context():
    if not AMP_ENABLED:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        except TypeError:
            return torch.amp.GradScaler(enabled=AMP_ENABLED)
    return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)


def release_cuda_cache_if_needed(usage_threshold: float = 0.92) -> None:
    if not CUDA_AVAILABLE:
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_ratio = 1.0 - (free_bytes / max(1, total_bytes))
        if used_ratio >= usage_threshold:
            torch.cuda.empty_cache()
    except Exception:
        pass


def print_runtime_profile(profile: Dict[str, Any]) -> None:
    print("Runtime Profile:")
    print(f"  - logical_cpu_count: {profile['logical_cpu_count']}")
    print(f"  - physical_core_estimate: {profile['physical_core_estimate']}")
    print(f"  - recommended_xgb_n_jobs: {profile['recommended_xgb_n_jobs']}")
    print(f"  - recommended_joblib_cpus: {profile['recommended_joblib_cpus']}")
    print(f"  - recommended_torch_threads: {profile['recommended_torch_threads']}")
    print(f"  - recommended_dataloader_workers: {profile['recommended_dataloader_workers']}")
    print(f"  - recommended_inference_batch_size: {profile['recommended_inference_batch_size']}")
    print(f"  - recommended_xgb_device: {profile['recommended_xgb_device']}")
    print(f"  - pin_memory: {profile['pin_memory']}")
    print(f"  - amp_enabled: {profile['amp_enabled']}")
    print(f"  - cuda_available: {profile['cuda_available']}")
    if profile["cuda_available"]:
        print(f"  - gpu_name: {profile['gpu_name']}")
        print(f"  - gpu_memory_gb: {profile['gpu_memory_gb']}")
        print(f"  - tf32_matmul: {profile['tf32_matmul']}")
        print(f"  - tf32_cudnn: {profile['tf32_cudnn']}")
        print(f"  - cudnn_benchmark: {profile['cudnn_benchmark']}")
