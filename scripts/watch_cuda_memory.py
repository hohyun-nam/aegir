#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from datetime import datetime

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor memory of the CUDA device visible inside this container."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    torch.cuda.set_device(args.device)
    total_gib = torch.cuda.get_device_properties(args.device).total_memory / 2**30
    while True:
        free, total = torch.cuda.mem_get_info(args.device)
        free_gib = free / 2**30
        used_gib = (total - free) / 2**30
        used_percent = 100.0 * (total - free) / total
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{stamp}  used={used_gib:6.2f} GiB  free={free_gib:6.2f} GiB  "
            f"total={total_gib:6.2f} GiB  usage={used_percent:5.1f}%",
            flush=True,
        )
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
