"""W8 Phase-1 evaluation: one inference, dual-set scores.

Hygiene (W8 D2 rules):
  - per-class thresholds chosen ONLY on the 583 cal split (max-F1, verbatim from
    scripts/w2_calibrate.py); the 1,623 report split is consumed once.
  - preprocessing / model / autocast identical to training (build_transforms +
    build_model_w8 + common.utils.autocast).
  - pipeline identity: 583 mAP recomputed here must match the run's last-epoch
    val_map within 1e-3, else abort.
  - strict state_dict load; checkpoint epoch asserted against --expect-stop-epoch.

Outputs (per run): thresholds.json, per_class_report.csv, eval_metrics.json.
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "scripts"))
from common.dataset import XrayDataset, build_transforms, read_labels, read_name_list  # noqa: E402
from common.metrics import compute_metrics  # noqa: E402
from common.utils import autocast, md5_file, save_json  # noqa: E402
from w8.train_phase1 import build_model_w8  # noqa: E402
from w2_calibrate import best_f1_threshold  # noqa: E402

GATE_MAP = 0.56
GATE_F1 = 0.52
GATE_MILD = 0.38
GATE_REF = 0.28
GATE_SIX = {"NOF": 0.8337, "PLE": 0.8439, "CMG": 0.7212,
            "ALV": 0.7007, "PTX": 0.6992, "FB": 0.6000}
FALLBACK_ABBR = {
    "no_finding": "NOF", "cardiomegaly": "CMG", "alveolar_pattern": "ALV",
    "bronchial_pattern": "BRC", "pleural_effusion": "PLE", "mass": "MAS",
    "interstitial_pattern": "INT", "pneumothorax": "PTX",
    "pleural_mineralization": "PLM", "megaesophagus": "MGE",
    "foreign_body": "FB", "hernia": "HRN",
}


def parse_args():
    p = argparse.ArgumentParser("W8 eval: 583 calibration + 1,623 report, one inference per set")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="convnext_small")
    p.add_argument("--labels", default=r"<VETPETS_ROOT>\data\labels.csv")
    p.add_argument("--images-dir", default=r"<VETPETS_ROOT>\data\images")
    p.add_argument("--cal-file", default=r"<VETPETS_ROOT>\outputs\W2\splits\cal.csv")
    p.add_argument("--report-file", default=r"<VETPETS_ROOT>\data\val.csv")
    p.add_argument("--metrics-json", default=None,
                   help="training metrics.json for the 583 identity check")
    p.add_argument("--expect-stop-epoch", type=int, required=True)
    p.add_argument("--model-card", default=r"<VETPETS_ROOT>\outputs\W3\model_card\model_card.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets = [], []
    for x, y, _ in loader:
        x = x.to("cuda", non_blocking=True)
        with autocast():
            out = model(x)
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "eval_log.txt"), "w", encoding="utf-8")

    def L(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    assert ck.get("model_name") == a.model, f"ckpt model {ck.get('model_name')} != {a.model}"
    assert ck.get("consumable", True), "checkpoint marked non-consumable (probe)"
    assert int(ck["epoch"]) == a.expect_stop_epoch, (
        f"stop epoch mismatch: ckpt {ck['epoch']} != expected {a.expect_stop_epoch}")
    L(f"[ckpt] {a.ckpt} epoch={ck['epoch']} val_map={ck.get('val_map')}")

    df, label_classes = read_labels(a.labels)
    assert set(classes) <= set(label_classes), "checkpoint classes missing from labels.csv"
    order = [c for c in label_classes if c in classes]
    assert order == classes, "checkpoint class order != labels.csv relative order"

    cal_names = read_name_list(a.cal_file)
    rep_names = read_name_list(a.report_file)
    assert not (set(cal_names) & set(rep_names)), "cal/report overlap"
    L(f"[data] cal={len(cal_names)} report={len(rep_names)}")

    model = build_model_w8(a.model, len(classes), pretrained=False).to("cuda")
    model.load_state_dict(ck["model"], strict=True)
    L("[model] strict load OK")

    def loader(names):
        ds = XrayDataset(a.images_dir, names, classes, df, build_transforms(a.size, train=False))
        return DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                          num_workers=a.workers, pin_memory=True)

    cal_p, cal_y = predict(model, loader(cal_names))
    rep_p, rep_y = predict(model, loader(rep_names))

    # ---- pipeline identity: 583 mAP/F1 vs training log (tolerance 1e-3)
    cal_met = compute_metrics(cal_y, cal_p, classes, threshold=0.5)
    identity = {"cal_macro_map": float(cal_met["macro_map"]),
                "cal_macro_f1_at_0.5": float(cal_met["macro_f1"])}
    if a.metrics_json:
        m = json.load(open(a.metrics_json, encoding="utf-8"))
        ref_map = float(m["epochs"][-1]["val_map"])
        ref_f1 = float(m["epochs"][-1]["val_f1"])
        d_map = abs(identity["cal_macro_map"] - ref_map)
        d_f1 = abs(identity["cal_macro_f1_at_0.5"] - ref_f1)
        identity.update({"train_log_val_map": ref_map, "abs_diff_map": d_map,
                         "train_log_val_f1": ref_f1, "abs_diff_f1": d_f1,
                         "ok": d_map <= 1e-3 and d_f1 <= 1e-3})
        L(f"[identity] cal mAP {identity['cal_macro_map']:.6f} vs {ref_map:.6f} d={d_map:.2e}; "
          f"cal F1 {identity['cal_macro_f1_at_0.5']:.6f} vs {ref_f1:.6f} d={d_f1:.2e} "
          f"{'OK' if identity['ok'] else 'FAIL'}")
        if not identity["ok"]:
            raise SystemExit("pipeline identity check failed (>1e-3)")

    # ---- thresholds on cal only
    thrs = {}
    for i, c in enumerate(classes):
        cal_pos = int(cal_y[:, i].sum())
        if cal_pos == 0:
            thrs[c] = {"threshold": 0.5, "status": "default 0.5 (no cal positives)", "cal_pos": 0}
        else:
            t, f, s = best_f1_threshold(cal_y[:, i], cal_p[:, i])
            thrs[c] = {"threshold": t, "status": s, "cal_pos": cal_pos, "cal_f1": f}

    # ---- report metrics (1,623), consumed once
    abbr = dict(FALLBACK_ABBR)
    if os.path.exists(a.model_card):
        for cc in json.load(open(a.model_card, encoding="utf-8"))["classes"]:
            if cc.get("class") and cc.get("abbr"):
                abbr[cc["class"]] = cc["abbr"]

    rows, aps = [], {}
    for i, c in enumerate(classes):
        pos = int(rep_y[:, i].sum())
        ap = float(average_precision_score(rep_y[:, i], rep_p[:, i])) if pos else float("nan")
        aps[c] = ap
        pred = (rep_p[:, i] >= thrs[c]["threshold"]).astype(int)
        tp = int(((rep_y[:, i] == 1) & (pred == 1)).sum())
        fp = int(((rep_y[:, i] == 0) & (pred == 1)).sum())
        fn = int(((rep_y[:, i] == 1) & (pred == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"class": c, "abbr": abbr.get(c, c), "report_pos": pos,
                     "threshold": thrs[c]["threshold"], "threshold_status": thrs[c]["status"],
                     "AP": ap, "F1": f1, "P": p, "R": r})
        L(f"[report] {abbr.get(c, c):>4} pos={pos:4d} thr={thrs[c]['threshold']:.4f} "
          f"AP={ap:.4f} F1={f1:.4f}")

    ev = [r for r in rows if r["report_pos"] > 0]
    macro_map = float(np.nanmean([r["AP"] for r in ev]))
    macro_f1 = float(np.mean([r["F1"] for r in ev]))

    def gmean(group):
        vals = [r["AP"] for r in ev if r["abbr"] in group]
        return float(np.mean(vals)) if vals else float("nan")

    mild = gmean({"BRC", "MAS", "PLM"})
    ref = gmean({"INT", "MGE", "HRN"})
    six = {k: aps[next(c for c in classes if abbr.get(c) == k)] for k in GATE_SIX
           if any(abbr.get(c) == k for c in classes)}

    gates = {
        "map": {"value": macro_map, "gate": GATE_MAP, "pass": macro_map >= GATE_MAP,
                "near": GATE_MAP - macro_map <= 0.01},
        "f1": {"value": macro_f1, "gate": GATE_F1, "pass": macro_f1 >= GATE_F1,
               "near": GATE_F1 - macro_f1 <= 0.01},
        "mild_group": {"value": mild, "gate": GATE_MILD, "pass": mild >= GATE_MILD},
        "ref_group": {"value": ref, "gate": GATE_REF, "pass": ref >= GATE_REF},
        "nondegen_six": {k: {"value": six[k], "gate": v, "pass": six[k] >= v}
                         for k, v in GATE_SIX.items()},
    }
    evaluable_ok = (len(ev) == 12)

    out = {
        "model": a.model, "ckpt": a.ckpt, "ckpt_epoch": int(ck["epoch"]),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cal_file": a.cal_file, "cal_md5": md5_file(a.cal_file), "cal_n": len(cal_names),
        "report_file": a.report_file, "report_md5": md5_file(a.report_file),
        "report_n": len(rep_names),
        "identity": identity,
        "n_evaluable": len(ev), "n_evaluable_ok_12": evaluable_ok,
        "classes_evaluable": [r["class"] for r in ev],
        "macro_map": macro_map, "macro_f1": macro_f1,
        "mild_group_ap": mild, "ref_group_ap": ref,
        "gates": gates,
        "per_class": rows,
        "thresholds": thrs,
        "low_freq_note": "6 classes with 0 report positives excluded from macro; AP=gated none",
    }
    save_json(out, os.path.join(a.out_dir, "eval_metrics.json"))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "per_class_report.csv"),
                              index=False, encoding="utf-8-sig")
    save_json({"thresholds": thrs, "cal_md5": md5_file(a.cal_file)},
              os.path.join(a.out_dir, "thresholds.json"))
    L(f"[summary] mAP={macro_map:.4f} F1={macro_f1:.4f} mild={mild:.4f} "
      f"ref={ref:.4f} n_eval={len(ev)}")
    L("[summary] gates: " + " ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in gates.items()
        if isinstance(v, dict) and "pass" in v))
    log.close()


if __name__ == "__main__":
    main()
