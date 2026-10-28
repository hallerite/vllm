# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interrupt a native NCCL update and reload a replacement on the same server.

Requires the reload-abort lifecycle from vLLM PR #57854 in addition to the
recoverable communicator. Run only against a disposable server (GPU 1):

    VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES=1 \
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 vllm serve Qwen/Qwen3-0.6B \
        --enforce-eager --gpu-memory-utilization 0.08 --max-model-len 512 \
        --weight-transfer-config '{"backend":"nccl"}' --port 18402

Then, with GPU 0 available to the sender:

    python examples/rl/rlhf_nccl_recovery.py --packed

The control, --stock, deliberately strands a blocking communicator and requires
restarting the disposable server. Never use it against a shared serving engine.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

import httpx


def trainer(pipe, port, crash, packed, recoverable, model_path):
    import torch
    from safetensors.torch import load_file

    from vllm.distributed.weight_transfer.nccl_common import (
        stateless_init_process_group,
    )
    from vllm.distributed.weight_transfer.packed_tensor import (
        packed_nccl_broadcast_producer,
    )

    torch.accelerator.set_device_index(0)
    torch.cuda.set_per_process_memory_fraction(0.08)
    state = {}
    for file in Path(model_path).glob("*.safetensors"):
        state.update(load_file(file))
    state = dict(sorted(state.items()))
    pipe.send(
        {
            "names": list(state),
            "dtype_names": [str(t.dtype).split(".")[-1] for t in state.values()],
            "shapes": [list(t.shape) for t in state.values()],
        }
    )
    comm = stateless_init_process_group(
        "127.0.0.1", port, 0, 2, 0, **({"timeout": 8} if recoverable else {})
    )
    pipe.send("initialized")
    pipe.recv()
    wait = comm.synchronize if recoverable else lambda x: x.synchronize()
    if crash:
        first = next(iter(state.values())).cuda().zero_()
        assert first.numel() * first.element_size() > 8 * 1024**2
        comm.broadcast(first.view(torch.uint8).flatten() if packed else first, 0)
        wait(torch.cuda.current_stream())
        pipe.send("partial_weights_sent")
        pipe.recv()
    else:
        if packed:
            packed_nccl_broadcast_producer(
                iter(state.items()),
                comm,
                0,
                lambda item: item[1].cuda(),
                buffer_size_bytes=8 * 1024**2,
                synchronize=wait if recoverable else None,
            )
        else:
            for tensor in state.values():
                send = tensor.cuda()
                comm.broadcast(send, 0)
                wait(torch.cuda.current_stream())
        pipe.send("complete")
        pipe.recv()
    comm.destroy()


async def recv(pipe):
    assert await asyncio.to_thread(pipe.poll, 60), "sender did not respond"
    return pipe.recv()


async def main(args):
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        "Qwen/Qwen3-0.6B", allow_patterns=["*.safetensors", "config.json"]
    )
    processes = []
    ctx = mp.get_context("spawn")
    async with httpx.AsyncClient(base_url="http://127.0.0.1:18402", timeout=90) as c:

        async def post(path, data=None):
            r = await c.post(path, json=data)
            r.raise_for_status()
            return r

        async def completion():
            r = await post(
                "/v1/completions",
                {
                    "model": "Qwen/Qwen3-0.6B",
                    "prompt": "The capital of France is",
                    "temperature": 0,
                    "max_tokens": 12,
                    "seed": 0,
                },
            )
            return r.json()["choices"][0]["text"]

        before = await completion()
        print(json.dumps({"event": "before", "text": before}), flush=True)
        await post("/shutdown_weight_transfer_engine")
        await post("/pause?mode=wait")
        try:
            for failure in (True, False):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                parent, child = ctx.Pipe()
                p = ctx.Process(
                    target=trainer,
                    args=(
                        child,
                        port,
                        failure,
                        args.packed,
                        not args.stock,
                        model_path,
                    ),
                )
                processes.append(p)
                p.start()
                meta = await recv(parent)
                init = {
                    "master_address": "127.0.0.1",
                    "master_port": port,
                    "rank_offset": 1,
                    "world_size": 2,
                    "packed": args.packed,
                    "packed_buffer_size_bytes": 8 * 1024**2,
                }
                if not args.stock:
                    init["operation_timeout"] = 8
                await post("/init_weight_transfer_engine", {"init_info": init})
                assert await recv(parent) == "initialized"
                await post("/start_weight_update")
                update = asyncio.create_task(
                    c.post("/update_weights", json={"update_info": meta})
                )
                parent.send("send")
                if failure:
                    assert await recv(parent) == "partial_weights_sent"
                    start = time.monotonic()
                    p.kill()
                    await asyncio.to_thread(p.join, 10)
                    try:
                        response = await asyncio.wait_for(update, 20)
                    except TimeoutError:
                        print(
                            "FAIL: interrupted transfer still blocked after 20s",
                            flush=True,
                        )
                        raise
                    assert response.status_code == 500, response.text
                    print(
                        json.dumps(
                            {
                                "event": "failed_transfer_returned",
                                "seconds": time.monotonic() - start,
                                "response": response.text[:500],
                            }
                        ),
                        flush=True,
                    )
                    assert (await c.get("/is_paused")).json()["is_paused"]
                    await post("/shutdown_weight_transfer_engine")
                else:
                    (await update).raise_for_status()
                    assert await recv(parent) == "complete"
                    await post(
                        "/finish_weight_update", {"weight_version": "replacement"}
                    )
                    await post("/resume")
                    after = await completion()
                    assert after == before, (before, after)
                    assert (await c.get("/weight_info")).json()[
                        "weight_version"
                    ] == "replacement"
                    parent.send("close")
                    await asyncio.to_thread(p.join, 10)
                    assert p.exitcode == 0
                    await post("/shutdown_weight_transfer_engine")
                    print(
                        json.dumps(
                            {"event": "PASS", "packed": args.packed, "text": after}
                        ),
                        flush=True,
                    )
        finally:
            for p in processes:
                if p.is_alive():
                    p.kill()
                p.join(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--stock", action="store_true")
    args = parser.parse_args()
    os.environ.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1", OMP_NUM_THREADS="2")
    asyncio.run(main(args))
