import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session", autouse=True)
def dist_process_group():
    """Single-process NCCL group for the whole test session -- everything
    under muon_research.scripts/optim assumes torch.distributed is already
    initialized (Geon.step, DistributedDataCursor, etc.), same as under a
    real (possibly single-GPU) torchrun launch."""
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for these tests (matches production: "
            "the training/fork code is CUDA-only)"
        )
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl", world_size=1, rank=0, device_id=torch.device("cuda", 0)
    )
    yield
    dist.destroy_process_group()
