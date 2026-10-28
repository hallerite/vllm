# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nonblocking NCCL communicator for recoverable dense weight transfers."""

import ctypes
import math
import os
import time
from typing import TYPE_CHECKING, cast

import torch

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclUniqueId,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup


class _Config(ctypes.Structure):
    # NCCL's versioned 2.17 ABI; newer libraries default the absent fields.
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int),
        ("cgaClusterSize", ctypes.c_int),
        ("minCTAs", ctypes.c_int),
        ("maxCTAs", ctypes.c_int),
        ("netName", ctypes.c_char_p),
    ]


class NCCLWeightTransferCommunicator:
    """A single-owner communicator, never captured in inference CUDA graphs.

    Broadcasts enqueue asynchronously. Call ``synchronize`` before consuming a
    received tensor on the CPU or releasing buffers. Both launch and completion
    waits poll for errors; failure aborts the communicator on the calling thread.
    """

    def __init__(
        self,
        unique_id: ncclUniqueId,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: float,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("NCCL operation timeout must be finite and positive")
        if os.environ.get("NCCL_COMM_BLOCKING") not in (None, "0"):
            raise ValueError("NCCL_COMM_BLOCKING must be unset or 0 for recovery")
        self.device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else device
        )
        if not current_platform.is_cuda() or self.device.type != "cuda":
            raise ValueError("Recoverable NCCL weight transfer requires CUDA")
        self.group: StatelessProcessGroup | None = None
        self.rank = rank
        self.world_size = world_size
        self.timeout = timeout
        self.nccl = NCCLLibrary()
        self.comm = ncclComm_t()
        self._init = self.nccl.lib.ncclCommInitRankConfig
        self._init.restype = ctypes.c_int
        self._init.argtypes = [
            ctypes.POINTER(ncclComm_t),
            ctypes.c_int,
            ncclUniqueId,
            ctypes.c_int,
            ctypes.POINTER(_Config),
        ]
        self._async_error = self.nccl.lib.ncclCommGetAsyncError
        self._async_error.restype = ctypes.c_int
        self._async_error.argtypes = [ncclComm_t, ctypes.POINTER(ctypes.c_int)]
        self._broadcast = self.nccl.lib.ncclBroadcast
        self._all_reduce = self.nccl.lib.ncclAllReduce
        undef = -(2**31)
        config = _Config(
            ctypes.sizeof(_Config), 0xCAFEBEEF, 21700, 0, undef, undef, undef, None
        )
        with torch.accelerator.device_index(self.device.index):
            try:
                self._wait_launch(
                    self._init(
                        ctypes.byref(self.comm), world_size, unique_id, rank, config
                    )
                )
                # Match PyNcclCommunicator's one-element float32 SUM handshake.
                data = torch.zeros(1, device=self.device)
                stream = torch.cuda.current_stream(self.device)
                self._wait_launch(
                    self._all_reduce(
                        buffer_type(data.data_ptr()),
                        buffer_type(data.data_ptr()),
                        1,
                        ncclDataTypeEnum.ncclFloat32,
                        0,
                        self.comm,
                        cudaStream_t(stream.cuda_stream),
                    )
                )
                self.synchronize(stream)
            except BaseException:
                self.destroy()
                raise

    def _check(self, result: int) -> None:
        if result not in (0, 7):  # ncclSuccess / ncclInProgress
            self.nccl.NCCL_CHECK(cast(ctypes.c_int, result))

    def _state(self) -> int:
        if not self.comm:
            raise RuntimeError("NCCL weight communicator is closed; reinitialize it")
        state = ctypes.c_int()
        self.nccl.NCCL_CHECK(self._async_error(self.comm, ctypes.byref(state)))
        self._check(state.value)
        return state.value

    def _poll(self, start: float) -> None:
        elapsed = time.monotonic() - start
        if elapsed >= self.timeout:
            raise TimeoutError(f"NCCL weight transfer wait exceeded {self.timeout}s")
        if elapsed >= 0.001:
            time.sleep(0.0001)

    def _wait_launch(self, result: int) -> None:
        start = time.monotonic()
        self._check(result)
        while result == 7:
            result = self._state()
            if result == 7:
                self._poll(start)

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None) -> None:
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError("Weight broadcast needs a contiguous tensor on its device")
        with torch.accelerator.device_index(self.device.index):
            try:
                self._state()
                if stream is None:
                    stream = torch.cuda.current_stream(self.device)
                self._wait_launch(
                    self._broadcast(
                        buffer_type(tensor.data_ptr()),
                        buffer_type(tensor.data_ptr()),
                        tensor.numel(),
                        ncclDataTypeEnum.from_torch(tensor.dtype),
                        src,
                        self.comm,
                        cudaStream_t(stream.cuda_stream),
                    )
                )
            except BaseException:
                self.destroy()
                raise

    def synchronize(self, completion: torch.cuda.Stream | torch.cuda.Event) -> None:
        """Wait without entering an uninterruptible CUDA synchronization."""
        with torch.accelerator.device_index(self.device.index):
            try:
                start = time.monotonic()
                self._state()
                while not completion.query():
                    self._state()
                    self._poll(start)
                self._state()
            except BaseException:
                self.destroy()
                raise

    def destroy(self) -> None:
        if self.comm:
            with torch.accelerator.device_index(self.device.index):
                self.nccl.ncclCommAbort(self.comm)
            self.comm = ncclComm_t()
