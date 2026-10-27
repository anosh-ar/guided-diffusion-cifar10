"""
Compare several training runs (e.g. the three beta schedules): overlaid loss curves and a summary table.

Example:
    python compare_runs.py runs/*_linear runs/*_cosine runs/*_sigmoid --out assets/schedule_comparison
"""
import argparse
import json
from pathlib import Path

from ddpm.utils import plot_loss_curves


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--out", type=str, default="assets/schedule_comparison",
                   help="output prefix; writes <out>.png and <out>.md")
    p.add_argument("--label", type=str, default="schedule", help="config key used to label runs")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    curves, rows = {}, []
    for run in map(Path, args.runs):
        config = load_json(run / "config.json")
        if config is None or not (run / "metrics.csv").exists():
            print(f"skipping {run}: not a finished run directory")
            continue
        label = str(config.get(args.label, run.name))
        curves[label] = run / "metrics.csv"
        summary = load_json(run / "summary.json") or {}
        evaluation = load_json(run / "eval" / "results.json")
        best_fid = min(evaluation["results"], key=lambda r: r["fid"]) if evaluation else None
        rows.append((label, summary.get("best_val_loss"), summary.get("test_loss_ema"), best_fid))

    if not curves:
        raise SystemExit("No valid runs given.")

    plot_loss_curves(curves, out.with_suffix(".png"), title=f"Comparison by {args.label}")

    fmt = lambda v, spec: format(v, spec) if v is not None else "–"
    lines = [f"| {args.label} | best val loss | test loss (EMA) | best FID (at w) | IS at best FID |",
             "|---|---:|---:|---:|---:|"]
    for label, val, test, best in rows:
        fid_cell = f"{best['fid']:.2f} (w = {best['guidance_scale']:g})" if best else "–"
        is_cell = f"{best['is_mean']:.2f}" if best else "–"
        lines.append(f"| {label} | {fmt(val, '.5f')} | {fmt(test, '.5f')} | {fid_cell} | {is_cell} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out.with_suffix('.png')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
