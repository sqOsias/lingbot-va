#!/usr/bin/env python3
"""Render LingBot training_metrics.jsonl to a standalone SVG."""

import argparse
import html
import json
import math
from pathlib import Path


def points(rows, key, x0, y0, width, height, lo, hi):
    span = max(hi - lo, 1e-12)
    xmax = max(row["optimizer_step"] for row in rows)
    return " ".join(
        f"{x0 + width * row['optimizer_step'] / xmax:.2f},{y0 + height * (hi - math.log10(max(row[key], 1e-12))) / span:.2f}"
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No training metrics")
    values = [math.log10(max(row[key], 1e-12)) for row in rows for key in ("latent_loss", "action_loss")]
    lo, hi = min(values), max(values)
    if hi - lo < 0.2:
        lo, hi = lo - 0.1, hi + 0.1
    width, height = 1200, 620
    x0, y0, cw, ch = 85, 70, 1070, 455
    video = points(rows, "latent_loss", x0, y0, cw, ch, lo, hi)
    action = points(rows, "action_loss", x0, y0, cw, ch, lo, hi)
    ticks = []
    for i in range(6):
        y = y0 + ch * i / 5
        value = 10 ** (hi - (hi - lo) * i / 5)
        ticks.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+cw}" y2="{y:.2f}" stroke="#ddd"/><text x="{x0-10}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.4g}</text>')
    xmax = rows[-1]["optimizer_step"]
    for i in range(6):
        x = x0 + cw * i / 5
        ticks.append(f'<text x="{x:.2f}" y="{y0+ch+25}" text-anchor="middle" font-size="12">{round(xmax*i/5)}</text>')
    elapsed_hours = rows[-1]["elapsed_seconds"] / 3600
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<g font-family="Arial, sans-serif" fill="#222">
<text x="{x0}" y="34" font-size="22" font-weight="bold">{html.escape(args.title)}</text>
<text x="{x0}" y="54" font-size="12" fill="#555">{len(rows)} optimizer steps · {elapsed_hours:.2f} hours · logarithmic loss axis</text>
{''.join(ticks)}
<line x1="{x0}" y1="{y0+ch}" x2="{x0+cw}" y2="{y0+ch}" stroke="#333"/>
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ch}" stroke="#333"/>
<polyline points="{video}" fill="none" stroke="#2563eb" stroke-width="1.5"/>
<polyline points="{action}" fill="none" stroke="#dc2626" stroke-width="1.5"/>
<line x1="{x0}" y1="575" x2="{x0+35}" y2="575" stroke="#2563eb" stroke-width="2"/><text x="{x0+43}" y="579" font-size="13">video loss</text>
<line x1="{x0+160}" y1="575" x2="{x0+195}" y2="575" stroke="#dc2626" stroke-width="2"/><text x="{x0+203}" y="579" font-size="13">action loss</text>
<text x="{x0+cw/2}" y="558" text-anchor="middle" font-size="13">optimizer step</text>
<text x="22" y="{y0+ch/2}" transform="rotate(-90 22 {y0+ch/2})" text-anchor="middle" font-size="13">loss</text>
</g></svg>'''
    args.output.write_text(svg)
    print(json.dumps({"status": "PASS", "steps": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
