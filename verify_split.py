# -*- coding: utf-8 -*-
"""Verify the released split lists against each other and, optionally, the dataset manifest.

Checks
------
1. row counts: train 5,489 / calibration 583 / reporting 1,623 (sum 7,695)
2. no case_id appears in more than one list
3. no patient_id appears in more than one list (patient-level separation)
4. no duplicate case_id inside a list
5. (optional) the union matches the case_id set of a dataset manifest

Usage
-----
    python verify_split.py
    python verify_split.py --manifest full_cases_7695.csv

The manifest is the case-level CSV distributed with the dataset
(DOI 10.5281/zenodo.19051776); the expected columns are ``case_id`` and
``split`` (``train`` / ``val``). Exit code 0 = all checks passed.
"""
import argparse
import csv
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {"train": 5489, "calibration": 583, "reporting": 1623}


def load(name):
    path = os.path.join(BASE, "data", "splits", f"{name}.csv")
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="case-level manifest CSV (case_id, split)")
    args = ap.parse_args()

    ok = True
    lists = {k: load(k) for k in EXPECT}

    for name, rows in lists.items():
        n = len(rows)
        good = n == EXPECT[name]
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {n} rows (expected {EXPECT[name]})")

    total = sum(len(v) for v in lists.values())
    good = total == 7695
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] total: {total} (expected 7695)")

    ids = {k: [r["case_id"] for r in v] for k, v in lists.items()}
    for name, v in ids.items():
        good = len(v) == len(set(v))
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: unique case_id")

    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(ids[names[i]]) & set(ids[names[j]])
            good = not inter
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] {names[i]} ∩ {names[j]} = {len(inter)}")

    pats = {k: set(r["patient_id"] for r in v) for k, v in lists.items()}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = pats[names[i]] & pats[names[j]]
            good = not inter
            ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] patient overlap {names[i]}/{names[j]} = {len(inter)}")

    if args.manifest:
        with open(args.manifest, encoding="utf-8-sig") as f:
            man = {r["case_id"]: r["split"] for r in csv.DictReader(f)}
        good = len(man) == 7695
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] manifest rows = {len(man)} (expected 7695)")

        union = set(ids["train"]) | set(ids["calibration"]) | set(ids["reporting"])
        good = union == set(man)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] union == manifest case_id set "
              f"(missing {len(set(man) - union)}, extra {len(union - set(man))})")

        for name in ("train", "calibration"):
            bad = [c for c in ids[name] if man.get(c) != "train"]
            ok &= not bad
            print(f"[{'PASS' if not bad else 'FAIL'}] {name} all inside manifest split=train ({len(bad)} off)")
        bad = [c for c in ids["reporting"] if man.get(c) != "val"]
        ok &= not bad
        print(f"[{'PASS' if not bad else 'FAIL'}] reporting all inside manifest split=val ({len(bad)} off)")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
