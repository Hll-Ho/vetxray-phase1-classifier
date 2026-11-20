"""Evaluate a W1 checkpoint: val predictions CSV, metrics, PR / sample / Grad-CAM figures."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", r"<VETPETS_ROOT>\pretrained\torch")
os.environ.setdefault("HF_HOME", r"<VETPETS_ROOT>\pretrained\hf")

import argparse
import math
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.dataset import XrayDataset, build_transforms, read_labels, read_name_list
from common.metrics import LOW_FREQ_THRESHOLD, compute_metrics, low_frequency_classes
from common.utils import autocast, build_model, get_logger, save_json, seed_all
from common.viz import (ensure_class_map, plot_gradcam, plot_pr_curves,
                        plot_pred_samples)


def parse_args():
    p = argparse.ArgumentParser("W1 eval")
    p.add_argument("--model", required=True,
                   choices=["resnet50", "densenet121", "convnext_tiny", "vit_base_384"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data-dir", default=r"<VETPETS_ROOT>\data")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--val-file", required=True)
    p.add_argument("--out-root", default=r"<VETPETS_ROOT>\outputs\W1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            out = model(x)
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def _resize_cam(cam, size):
    cam = np.maximum(cam, 0)
    mx = float(cam.max())
    cam = cam / mx if mx > 0 else np.zeros_like(cam)
    t = torch.from_numpy(cam.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _gradcam_cnn(model, x, cls, layer):
    acts, grads = [], []

    def fwd(_m, _i, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g))

    handle = layer.register_forward_hook(fwd)
    model.zero_grad(set_to_none=True)
    out = model(x)
    out[0, cls].backward()
    handle.remove()
    act, grad = acts[0][0], grads[0][0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    return cam.detach().float().cpu().numpy()


def _gradcam_vit(model, x):
    attns, handles, fused_old = [], [], []
    for blk in model.blocks:
        if hasattr(blk.attn, "fused_attn"):
            fused_old.append((blk.attn, blk.attn.fused_attn))
            blk.attn.fused_attn = False  # force explicit attention so attn_drop is called
        drop = blk.attn.attn_drop
        handles.append(drop.register_forward_hook(
            lambda _m, _i, out: attns.append(out.detach())))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        for module, old in fused_old:
            module.fused_attn = old
    n = attns[0].shape[-1]
    rollout = torch.eye(n, device=attns[0].device)
    for a in attns:
        mat = a[0].float().mean(dim=0)
        mat = mat + torch.eye(n, device=mat.device)
        mat = mat / mat.sum(dim=-1, keepdim=True)
        rollout = mat @ rollout
    cam = rollout[0, 1:]
    grid = int(round(math.sqrt(int(cam.numel()))))
    return cam.reshape(grid, grid).cpu().numpy()


def compute_cam(model, model_name, image_np, size, cls):
    rgb = np.stack([image_np, image_np, image_np], axis=-1)
    tensor = build_transforms(size, train=False)(image=rgb)["image"]
    if isinstance(tensor, np.ndarray):
        tensor = tensor.transpose(2, 0, 1)
    x = torch.from_numpy(np.ascontiguousarray(tensor)).float().unsqueeze(0).to("cuda")
    if model_name.startswith("vit"):
        cam = _gradcam_vit(model, x)
    else:
        if model_name == "resnet50":
            layer = model.layer4[-1]
        elif model_name == "densenet121":
            layer = model.features[-1]
        else:
            layer = model.features[-1]
        cam = _gradcam_cnn(model, x, cls, layer)
    return _resize_cam(cam, size)


def _display_image(path, size):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required")

    size = args.size or (384 if args.model.startswith("vit") else 512)
    if args.model.startswith("vit") and size != 384:
        raise SystemExit("vit_base_384 must run at 384; 224 fallback needs client approval")
    images_dir = args.images_dir or os.path.join(args.data_dir, "images")
    labels_path = args.labels or os.path.join(args.data_dir, "labels.csv")
    ckpt_path = args.ckpt or os.path.join(args.out_root, args.model, "best.pth")
    out_dir = os.path.join(args.out_root, args.model)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "eval_log.txt"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    if ckpt.get("model_name") and ckpt["model_name"] != args.model:
        raise SystemExit(f"checkpoint model={ckpt['model_name']} does not match --model={args.model}")
    logger.info("loaded %s | classes=%d | epoch=%s | val_map=%s",
                ckpt_path, len(classes), ckpt.get("epoch"), ckpt.get("val_map"))

    df, label_classes = read_labels(labels_path)
    if label_classes != classes:
        raise SystemExit("labels.csv class order != checkpoint class order; refusing to evaluate")
    n_pos_overall = {c: int(df[c].sum()) for c in classes}
    low_freq = low_frequency_classes(classes, n_pos_overall)
    val_names = read_name_list(args.val_file)
    missing = [n for n in val_names if n not in set(df["filename"])]
    if missing:
        raise SystemExit(f"{len(missing)} val filenames missing from labels.csv")

    ensure_class_map(classes, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "common", "class_map.csv"))
    val_ds = XrayDataset(images_dir, val_names, classes, df,
                         build_transforms(size, train=False))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    probs, targets = predict(model, val_loader, device)
    met = compute_metrics(targets, probs, classes, threshold=args.threshold)

    pred_csv = os.path.join(args.out_root, f"{args.model}_val_pred.csv")
    out_df = pd.DataFrame({"filename": val_names})
    for i, c in enumerate(classes):
        out_df[f"prob_{c}"] = probs[:, i]
        out_df[f"pred_{c}"] = (probs[:, i] >= args.threshold).astype(int)
    out_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    logger.info("val n=%d macro_mAP=%.4f macro_F1=%.4f (%s)", met["n_eval"],
                met["macro_map"], met["macro_f1"], met["macro_map_note"])
    logger.info("contradictions (pred): %d/%d = %.4f | label-side: %d/%d = %.4f",
                met["n_contradiction"], met["n_eval"], met["contradiction_rate"],
                met["n_label_contradiction"], met["n_eval"], met["label_contradiction_rate"])
    logger.info("low-frequency classes (<%d positives overall, M3): %s",
                LOW_FREQ_THRESHOLD, low_freq or "none")
    for c in classes:
        ap = met["per_class_ap"][c]
        logger.info("  %-28s AP=%s F1=%.3f n_pos=%d", c,
                    "n/a" if np.isnan(ap) else f"{ap:.3f}",
                    met["per_class_f1"][c], met["n_pos"][c])

    plot_pr_curves(targets, probs, classes, met["per_class_ap"],
                   os.path.join(figs_dir, "pr_curves.png"))

    # images are loaded lazily for the selected sample indices only
    n = len(val_names)
    rng = np.random.default_rng(args.seed)
    rand_idx = sorted(int(i) for i in
                      rng.choice(n, size=min(8, n), replace=False)) if n else []
    err = np.abs(probs - targets).mean(axis=1)
    worst_idx = [int(i) for i in np.argsort(-err)[:min(8, n)]]
    needed = sorted(set(rand_idx) | set(worst_idx) | set(worst_idx[:4]))
    images = {i: _display_image(os.path.join(images_dir, val_names[i]), size) for i in needed}
    plot_pred_samples(images, val_names, targets, probs, classes, rand_idx,
                      os.path.join(figs_dir, "pred_samples.png"),
                      title="Random val samples (hit=green, FP=red, miss=gray)",
                      threshold=args.threshold)
    plot_pred_samples(images, val_names, targets, probs, classes, worst_idx,
                      os.path.join(figs_dir, "pred_worst.png"),
                      title="Worst 8 val samples by mean |prob - target|",
                      threshold=args.threshold)

    gradcam_status, cams = "ok", []
    cam_idx = worst_idx[:4]
    try:
        for idx in cam_idx:
            cls = int(np.argmax(probs[idx]))
            cams.append(compute_cam(model, args.model, images[idx], size, cls))
        plot_gradcam(images, cams, val_names, targets, probs, classes, cam_idx,
                     os.path.join(figs_dir, "gradcam.png"), threshold=args.threshold)
        skipped_path = os.path.join(figs_dir, "gradcam_skipped.txt")
        if os.path.exists(skipped_path):
            os.remove(skipped_path)
    except Exception as e:  # best effort per W1a A2
        gradcam_status = f"skipped: {type(e).__name__}: {e}"
        logger.warning("Grad-CAM skipped: %s", gradcam_status)
        with open(os.path.join(figs_dir, "gradcam_skipped.txt"), "w", encoding="utf-8") as f:
            f.write(gradcam_status + "\n")

    save_json({
        "model": args.model,
        "ckpt": ckpt_path,
        "epoch": ckpt.get("epoch"),
        "size": size,
        "threshold": args.threshold,
        "macro_map": None if np.isnan(met["macro_map"]) else met["macro_map"],
        "macro_f1": None if np.isnan(met["macro_f1"]) else met["macro_f1"],
        "macro_map_note": met["macro_map_note"],
        "n_evaluable": met["n_evaluable"],
        "contradiction_rate": (None if np.isnan(met["contradiction_rate"])
                               else met["contradiction_rate"]),
        "n_contradiction": met["n_contradiction"],
        "label_contradiction_rate": (None if np.isnan(met["label_contradiction_rate"])
                                     else met["label_contradiction_rate"]),
        "n_label_contradiction": met["n_label_contradiction"],
        "per_class_ap": {k: (None if np.isnan(v) else float(v))
                         for k, v in met["per_class_ap"].items()},
        "per_class_ap_status": met["per_class_ap_status"],
        "per_class_f1": met["per_class_f1"],
        "per_class_precision": met["per_class_precision"],
        "per_class_recall": met["per_class_recall"],
        "n_pos": met["n_pos"],
        "n_eval": met["n_eval"],
        "low_freq_classes": low_freq,
        "low_freq_threshold": LOW_FREQ_THRESHOLD,
        "classes": classes,
        "gradcam": gradcam_status,
        "val_pred_csv": pred_csv,
    }, os.path.join(out_dir, "eval_metrics.json"))
    logger.info("eval done: %s | gradcam=%s", pred_csv, gradcam_status)


if __name__ == "__main__":
    main()
