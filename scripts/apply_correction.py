"""Which results survive a multiple-comparison correction.

Testing 14 correlated metrics at alpha=0.05 finds something on noise alone
roughly half the time. Benjamini-Hochberg controls the expected proportion of
reported discoveries that are false, which is the right control for a table of
related retrieval metrics; Bonferroni controls the chance of any false positive
at all and is needlessly severe when P@5 and P@10 move together by construction.

judged@k is excluded from the family. In this eval every retrieved document is
judged, so judged@k is numerically identical to P@k, and counting both would
inflate the number of tests with an exact duplicate.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medcase.evaluate import benjamini_hochberg

TEST = "p_bootstrap"  # the paired test; p_sign ignores effect size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--file", default="data/eval_results.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / args.file).read_text())
    out = {}

    for arm, cmp in data.get("comparisons", {}).items():
        metrics = {
            name: entry
            for name, entry in cmp.get("metrics", {}).items()
            if not name.startswith("judged@") and entry.get(TEST) is not None
        }
        pvals = {n: e[TEST] for n, e in metrics.items()}
        bh = benjamini_hochberg(pvals, alpha=args.alpha)
        bonf = {n: min(1.0, p * len(pvals)) for n, p in pvals.items()}

        print(f"\n=== {arm} vs faiss-baseline ===")
        print(f"{len(pvals)} tests, alpha {args.alpha}, "
              f"Bonferroni threshold {args.alpha/len(pvals):.4f}\n")
        print(f"{'metric':<10}{'delta':>9}{'p(boot)':>10}{'BH q':>9}"
              f"{'BH':>5}{'Bonf':>6}")
        print("-" * 49)
        rows = sorted(metrics.items(), key=lambda kv: kv[1][TEST])
        for name, entry in rows:
            r = bh["results"][name]
            print(f"{name:<10}{entry['delta']:>+9.4f}{entry[TEST]:>10.4f}"
                  f"{r['q']:>9.4f}{'yes' if r['reject'] else 'no':>5}"
                  f"{'yes' if bonf[name] <= args.alpha else 'no':>6}")
        print(f"\nsurvive BH: {bh['n_significant']}/{len(pvals)}")
        out[arm] = {
            "n_tests": len(pvals),
            "alpha": args.alpha,
            "benjamini_hochberg": bh,
            "bonferroni_significant": [
                n for n, v in bonf.items() if v <= args.alpha
            ],
        }

    path = root / "data" / "correction_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
