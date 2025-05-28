#!/usr/bin/env python
"""
flowmm_dataset_builder.py

Create FlowMM-ready CSV splits (train / val / test).

Dependencies
------------
pip install "jarvis-tools>=2024.5" "pymatgen>=2024.1" pandas numpy tqdm

Usage
-----
python flowmm_dataset_builder.py \
    --dataset dft_3d \
    --id-key jid \
    --target Tc_supercon \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --seed 123
"""

import argparse, random, json, numpy as np, pandas as pd
from tqdm import tqdm

from jarvis.db.figshare import data as jarvis_data
from jarvis.core.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def canonicalise(pmg_struct: Structure, symprec: float = 0.1):
    """Return canonical cell CIF + (spg, spg_conv)."""
    sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec)
    spg_num = sga.get_space_group_number()
    conv = sga.get_conventional_standard_structure()
    spg_conv = SpacegroupAnalyzer(conv, symprec=symprec).get_space_group_number()
    return conv.to(fmt="cif"), spg_num, spg_conv


def make_dataframe(dataset_name: str, id_key: str, target_key: str):
    records = []
    for item in tqdm(jarvis_data(dataset_name), desc="Downloading/JARVIS"):
        if item[target_key] == "na":
            continue

        pmg = Atoms.from_dict(item["atoms"]).pymatgen_converter()

        try:
            cif_raw = pmg.to(fmt="cif")
            cif_conv, spg, spg_conv = canonicalise(pmg)

            rec = {
                "material_id": item[id_key],
                "pretty_formula": pmg.composition.reduced_formula,
                "elements": json.dumps([el.symbol for el in pmg.species]),
                "cif": cif_raw,
                "spacegroup.number": spg,
                "spacegroup.number.conv": spg_conv,
                "cif.conv": cif_conv,
                # the target property is optional for FlowMM
                target_key: item[target_key],
            }
            records.append(rec)
        except Exception as exc:  # bad CIF or symmetry failure
            print(f"Skipping {item[id_key]}: {exc}")

    return pd.DataFrame(records)


def split_indices(n, train_ratio, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    id_train = idx[:n_train]
    id_val = idx[n_train : n_train + n_val]
    id_test = idx[n_train + n_val :]
    return id_train, id_val, id_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="JARVIS dataset name, e.g. dft_3d, dft_2d, c2db")
    ap.add_argument("--id-key", default="jid", help="Field to use as material_id")
    ap.add_argument("--target", dest="target_key", default="Tc_supercon",
                    help="Scalar property column (kept for bookkeeping)")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    assert abs(
        args.train_ratio + args.val_ratio + args.test_ratio - 1
    ) < 1e-6, "splits must sum to 1"

    df = make_dataframe(args.dataset, args.id_key, args.target_key)
    id_train, id_val, id_test = split_indices(
        len(df), args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    df.iloc[id_train].to_csv("train.csv", index=False)
    df.iloc[id_val].to_csv("val.csv", index=False)
    df.iloc[id_test].to_csv("test.csv", index=False)

    print("✓ Wrote train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
