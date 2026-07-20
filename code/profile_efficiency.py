"""profile_efficiency.py  --  Reviewer 1 (Comment 14), Reviewer 2 (Comment 4)

Reports parameter count, GMACs/GFLOPs, single-image latency, batched throughput
and peak GPU memory for ABSS-Net on the current device. Run on the A100 used for
training and paste the REAL numbers into the Computational Efficiency subsection.
"""
import time
import torch


def profile(model, device="cuda", img_size=224, warmup=10, iters=50, batch=32):
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters      : {n:,} ({n / 1e6:.2f} M)")
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(
            model, inputs=(torch.randn(1, 3, img_size, img_size).to(device),),
            verbose=False)
        print(f"Compute         : {macs / 1e9:.3f} GMACs | {2 * macs / 1e9:.3f} GFLOPs")
    except Exception as e:  # pip install thop
        print("thop unavailable:", e)

    def _bench(bs):
        x = torch.randn(bs, 3, img_size, img_size).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        return (time.time() - t0) / iters

    lat = _bench(1)
    print(f"Latency (bs=1)  : {lat * 1e3:.2f} ms/image | {1 / lat:.1f} img/s")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    tb = _bench(batch)
    print(f"Throughput      : {batch / tb:.1f} img/s (bs={batch})")
    if device.startswith("cuda"):
        print(f"Peak GPU memory : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    raise SystemExit(
        "profile_efficiency.py exposes profile(model, device, ...). "
        "Import this function after constructing/loading ABSSNet; a trained model object is required."
    )
