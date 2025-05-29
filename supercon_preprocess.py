#!/usr/bin/env python
"""
supercon_preprocess.py  –  Python 3.9 compatible

Example
-------
python supercon_preprocess.py \
    --dataset dft_3d --id-key jid --target Tc_supercon \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --seed 123 --max-size 1000
"""
from __future__ import annotations

import argparse, random, json, hashlib
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from jarvis.db.figshare import data as jarvis_data
from jarvis.core.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ---------- helpers ----------------------------------------------------------
def canonicalise(pmg_struct: Structure, symprec: float = 0.1) -> Tuple[str, int, int]:
    """Return (cif_conv, spg_num, spg_num_conv).  Never raises."""
    try:
        sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec)
        spg_num = sga.get_space_group_number()
        conv = sga.get_conventional_standard_structure()
        spg_conv = SpacegroupAnalyzer(conv, symprec=symprec).get_space_group_number()
        return conv.to(fmt="cif"), spg_num, spg_conv
    except Exception:
        return "", -1, -1


def make_dataframe(
    dataset_name: str,
    id_key: str,
    target_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    """Download JARVIS records and keep those with a defined target.

    If --max-size is given, we **stop as soon as we have exactly that many
    valid structures** (i.e. after filtering out `"na"` / `None` targets).
    """
    records: List[dict] = []
    for item in tqdm(jarvis_data(dataset_name), desc="Downloading/JARVIS"):
        target_val = item.get(target_key, "na")
        if target_val in ("na", None):
            continue  # skip invalid target

        pmg = Atoms.from_dict(item["atoms"]).pymatgen_converter()
        try:
            cif_raw = pmg.to(fmt="cif")
        except Exception:
            continue  # bad CIF – behave like the reference script

        cif_conv, spg, spg_conv = canonicalise(pmg)

        records.append(
            {
                "material_id": item[id_key],
                "pretty_formula": pmg.composition.reduced_formula,
                "elements": json.dumps([el.symbol for el in pmg.species]),
                "cif": cif_raw,
                "spacegroup.number": spg,
                "spacegroup.number.conv": spg_conv,
                "cif.conv": cif_conv,
                target_key: target_val,
            }
        )

        # --- stop exactly at the requested cap -------------------------------
        if max_size is not None and len(records) == max_size:
            break

    return pd.DataFrame(records)


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Replicates the reference get_id_train_val_test()."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(train_ratio * n)
    n_test = int(test_ratio * n)
    n_val = int(val_ratio * n)

    if n_train + n_val + n_test > n:
        raise ValueError("Check total number of samples")

    id_train = indices[:n_train]
    id_val = indices[-(n_val + n_test) : -n_test]  # noqa: E203
    id_test = indices[-n_test:]
    return id_train, id_val, id_test


def sha(lst) -> str:
    m = hashlib.sha256()
    for x in lst:
        m.update(str(x).encode())
        m.update(b",")
    return m.hexdigest()[:10]


# ---------- CLI --------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--id-key", default="jid")
    ap.add_argument("--target", dest="target_key", default="Tc_supercon")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-size", type=int, default=None,
                    help="Cap on the number of *valid* structures")
    args = ap.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1) < 1e-6

    df = make_dataframe(args.dataset, args.id_key, args.target_key, args.max_size)
    print(f"Collected {len(df)} records (max-size={args.max_size})")

    id_train, id_val, id_test = split_indices(
        len(df), args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    Path(".").mkdir(exist_ok=True)
    df.iloc[id_train].to_csv("train.csv", index=False)
    df.iloc[id_val].to_csv("val.csv", index=False)
    df.iloc[id_test].to_csv("test.csv", index=False)

    print("✓ Wrote train.csv, val.csv, test.csv")
    print(f"hashes  train:{sha(df.iloc[id_train]['material_id'])} "
          f"val:{sha(df.iloc[id_val]['material_id'])} "
          f"test:{sha(df.iloc[id_test]['material_id'])}")


if __name__ == "__main__":
    main()
