"""
External runner: launches hinerv_main.py training (no pruning/quantization stages)
and reports PSNR at fixed wall-clock intervals, mirroring nika's wall_time_bench.py
so the two are directly comparable.

Usage:
    python wall_time_bench.py --config s --device 3 --batch_size 144 --interval_min 5 --max_min 22

Writes:
    - a raw passthrough log of the training stdout
    - a CSV with one row per interval: elapsed time, epoch reached, current train PSNR,
      GPU memory used (MiB)
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

EPOCH_END_RE = re.compile(
    r"^Train - Epoch (?P<epoch>\d+) \[(?P<step>\d+)/(?P<total>\d+)\]\s+"
    r"lr: (?P<lr>[\d.eE+-]+)\s+img/s: (?P<imgs>[\d.]+)\s+loss: (?P<loss>[\d.]+)\s+psnr: (?P<psnr>[\d.]+)$"
)
EVAL_BEST_RE = re.compile(
    r"^\*\*\*\s+psnr: (?P<psnr>[\d.]+)\s+ms-ssim: (?P<msssim>[\d.]+)\s+@ (?P<size>[\d.]+M)$"
)


def query_gpu_mem_mib(gpu_index):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Wall-clock PSNR reporter for hinerv_main.py (train-only, no prune/quant)")
    parser.add_argument("--interval_min", type=float, default=5.0, help="Report cadence in minutes")
    parser.add_argument("--max_min", type=float, default=None, help="Terminate training after this many wall-clock minutes")
    parser.add_argument("--out", default=None, help="CSV output path")
    parser.add_argument("--log", default=None, help="Raw stdout log path")

    parser.add_argument("--config", default="s", help="Bunny model size: xxs/xs/s/a-xxs/a-xs/a-s/b-xxs/b-xs/b-s")
    parser.add_argument("--train_cfg", default="hinerv_1280x720", help="Train cfg file (under cfgs/train, no .txt)")
    parser.add_argument("--dataset", default="data")
    parser.add_argument("--dataset_name", default="bunny")
    parser.add_argument("--device", default="0", help="Physical GPU index to pin via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch_size", type=int, default=144, help="Patch batch size (144 = one frame's worth of 80x80 patches)")
    parser.add_argument("--epochs", type=int, default=100000, help="Upper bound on epochs; wall-clock is the real stop condition")
    parser.add_argument("--eval_epochs", type=int, default=1_000_000,
                         help="Override hinerv_main.py's --eval-epochs. Default effectively disables in-loop eval "
                              "entirely (it's synchronous and would eat into the wall-clock training budget); "
                              "hinerv_main.py now saves an unconditional 'checkpoint_latest' every epoch regardless "
                              "of eval, so training stays pure and eval happens standalone afterward on whatever "
                              "was trained when the wall clock cut it off.")
    args = parser.parse_args()

    gpu_index = int(re.search(r"\d+", args.device).group())

    src_dir = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join(src_dir, f"walltime_bunny_{args.config}_{ts}.csv")
    log_path = args.log or os.path.join(src_dir, f"walltime_bunny_{args.config}_{ts}.log")
    output_dir = os.path.join("runs", f"walltime_{args.config}_{ts}")
    run_dir = os.path.join(src_dir, output_dir, "run")
    checkpoint_latest_dir = os.path.join(run_dir, "checkpoints", "checkpoint_latest")

    model_cfg_path = os.path.join(src_dir, "cfgs", "models", f"bunny-hinerv-{args.config}_1280x720.txt")
    train_cfg_path = os.path.join(src_dir, "cfgs", "train", f"{args.train_cfg}.txt")
    with open(model_cfg_path) as f:
        model_cfg_args = f.read().split()
    with open(train_cfg_path) as f:
        train_cfg_args = f.read().split()

    cmd = [
        sys.executable, "-u", "hinerv_main.py",
        "--dataset", args.dataset, "--dataset-name", args.dataset_name,
        "--output", output_dir, "--exp-name", "run",
    ] + train_cfg_args + model_cfg_args + [
        "--epochs", str(args.epochs),
        "--prune-epochs", "0",
        "--quant-epochs", "0",
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", "1",
        "--grad-accum", "1",
        "--log-eval", "false",
        "--seed", "0",
        "--eval-epochs", str(args.eval_epochs),
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print(f"Launching (CUDA_VISIBLE_DEVICES={gpu_index}): {' '.join(cmd)} (cwd={src_dir})")
    print(f"Raw log:        {log_path}")
    print(f"Wall-time CSV:  {out_path}")
    print(f"Reporting every {args.interval_min} min" + (f", max {args.max_min} min" if args.max_min else "") + "\n")

    interval_sec = args.interval_min * 60.0
    start = time.time()
    next_boundary = interval_sec
    latest_epoch = None
    latest_psnr = None
    best_eval_psnr = None
    stopped_early = False

    proc = subprocess.Popen(
        cmd, cwd=src_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    with open(log_path, "w") as logf, open(out_path, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["elapsed_seconds", "elapsed_minutes", "epoch", "train_psnr", "best_eval_psnr", "gpu_mem_used_mib"])
        outf.flush()

        try:
            for line in proc.stdout:
                now = time.time()
                elapsed = now - start
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)

                stripped = line.strip()
                m = EPOCH_END_RE.match(stripped)
                if m and m.group("step") == m.group("total"):
                    latest_epoch = int(m.group("epoch"))
                    latest_psnr = float(m.group("psnr"))
                else:
                    bm = EVAL_BEST_RE.match(stripped)
                    if bm:
                        best_eval_psnr = float(bm.group("psnr"))

                while elapsed >= next_boundary:
                    minutes = next_boundary / 60.0
                    mem_mib = query_gpu_mem_mib(gpu_index)
                    writer.writerow([f"{elapsed:.1f}", f"{minutes:.2f}", latest_epoch, latest_psnr, best_eval_psnr, mem_mib])
                    outf.flush()

                    # Snapshot whatever's on disk right now (the unconditional per-epoch
                    # checkpoint_latest save in hinerv_main.py) so this wall-clock mark has its
                    # own checkpoint to eval standalone later, independent of training continuing on.
                    snapshot_dir = os.path.join(run_dir, "checkpoints", f"checkpoint_{minutes:.0f}min")
                    if os.path.isdir(checkpoint_latest_dir) and not os.path.isdir(snapshot_dir):
                        try:
                            shutil.copytree(checkpoint_latest_dir, snapshot_dir)
                        except OSError as e:
                            # checkpoint_latest is being overwritten concurrently by the training
                            # process; a partial copy must never crash this monitor loop (that would
                            # skip the max_min check below and orphan the training subprocess).
                            print(f"    (snapshot copy failed, will be missing for this mark: {e!r})")
                            shutil.rmtree(snapshot_dir, ignore_errors=True)

                    print(f">>> [{minutes:.0f} min] epoch={latest_epoch} train_psnr={latest_psnr} "
                          f"(best_eval={best_eval_psnr}) gpu_mem={mem_mib}MiB snapshot={'yes' if os.path.isdir(snapshot_dir) else 'no'}")
                    next_boundary += interval_sec

                if args.max_min is not None and elapsed >= args.max_min * 60.0:
                    stopped_early = True
                    proc.terminate()
                    break
        except BaseException as e:
            # Any error here (not just Ctrl-C) must still terminate the child -- otherwise the
            # exception skips straight to the finally's proc.wait(), which blocks forever on an
            # orphaned, un-terminated training process (epochs is set to ~unbounded).
            stopped_early = True
            if not isinstance(e, KeyboardInterrupt):
                import traceback
                traceback.print_exc()
            proc.terminate()
        finally:
            proc.wait()
            total_elapsed = time.time() - start
            mem_mib = query_gpu_mem_mib(gpu_index)
            writer.writerow([f"{total_elapsed:.1f}", f"{total_elapsed/60:.2f}", latest_epoch, latest_psnr, best_eval_psnr, mem_mib])
            outf.flush()
            tag = "STOPPED (max_min reached)" if stopped_early else "Done"
            print(f"\n{tag}. Total elapsed: {total_elapsed/60:.2f} min. Final: "
                  f"epoch={latest_epoch} train_psnr={latest_psnr} best_eval={best_eval_psnr} gpu_mem={mem_mib}MiB")


if __name__ == "__main__":
    main()
