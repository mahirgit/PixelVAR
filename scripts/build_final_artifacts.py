#!/usr/bin/env python3
"""Build final PixelVAR decision tables and sample sheets from report metrics."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "reports" / "final"


def best_row(metrics_path: Path) -> dict:
    rows = list(csv.DictReader(metrics_path.open(newline="")))
    return min(rows, key=lambda row: float(row["feature_fid"]))


def fmt_top_k(value: str | None) -> str:
    return "none" if value in {"", None} else str(value)


def branches() -> list[dict]:
    return [
        {
            "rank": 1,
            "branch": "Real-only VAR",
            "run": "var_sprites_v0_full",
            "training_data": "Real MSD Sprites only",
            "metric_family": "palette-token feature score",
            "reference": "real Sprites validation",
            "comparable": "yes",
            "metrics": ROOT / "reports/eval/sprites_v0_full/metrics.csv",
            "grid": ROOT / "reports/eval/sprites_v0_full/temp_0.8_topk_8_grid.png",
            "decision": "WINNER / main result",
            "notes": "Best shared real-validation score; keep as primary checkpoint.",
        },
        {
            "rank": 2,
            "branch": "HMAR masked refinement",
            "run": "hmar_sprites_v0_full",
            "training_data": "Real MSD Sprites only",
            "metric_family": "palette-token feature score",
            "reference": "real Sprites validation",
            "comparable": "yes",
            "metrics": ROOT / "reports/eval/hmar_sprites_refinement_ablation/steps_1/metrics.csv",
            "grid": ROOT / "reports/eval/hmar_sprites_refinement_ablation/steps_1/temp_0.8_topk_8_grid.png",
            "decision": "Do not promote",
            "notes": "Technically successful; 1 refinement step is best, but still behind VAR.",
            "refinement_steps": "1",
        },
        {
            "rank": 3,
            "branch": "Generated-keep VAR",
            "run": "var_sprites_generated_keep_v0_full",
            "training_data": "Filtered 170k generated set",
            "metric_family": "palette-token feature score",
            "reference": "generated-keep validation",
            "comparable": "no",
            "metrics": ROOT / "reports/eval/sprites_generated_keep_v0_full/metrics.csv",
            "grid": ROOT / "reports/eval/sprites_generated_keep_v0_full/temp_0.8_topk_8_grid.png",
            "decision": "Do not promote",
            "notes": "Good self-reference score, but not the same real-validation comparison.",
        },
        {
            "rank": 4,
            "branch": "Real + generated mixed VAR",
            "run": "var_sprites_mixed_v0_full",
            "training_data": "Real Sprites + filtered generated set",
            "metric_family": "palette-token feature score",
            "reference": "real Sprites validation",
            "comparable": "yes",
            "metrics": ROOT / "reports/eval/sprites_mixed_v0_full_realval/metrics.csv",
            "grid": ROOT / "reports/eval/sprites_mixed_v0_full_realval/temp_0.8_topk_8_grid.png",
            "decision": "Do not promote",
            "notes": "Trains cleanly, but real-validation score is worse than real-only VAR.",
        },
        {
            "rank": 5,
            "branch": "OpenGameArt-mixed VAR",
            "run": "var_sprites_mixed_oga_v0_full",
            "training_data": "Real + generated + curated OpenGameArt",
            "metric_family": "palette-token feature score",
            "reference": "real Sprites validation",
            "comparable": "yes",
            "metrics": ROOT / "reports/eval/sprites_mixed_oga_v0_full_realval/metrics.csv",
            "grid": ROOT / "reports/eval/sprites_mixed_oga_v0_full_realval/temp_0.8_topk_8_grid.png",
            "decision": "Do not promote",
            "notes": "Public OGA pipeline works, but the added data does not improve quality.",
        },
        {
            "rank": 6,
            "branch": "Patch-VQ VAR",
            "run": "var_sprites_patchvq16_v0_full",
            "training_data": "Real Sprites encoded as learned 2x2 patch codes",
            "metric_family": "decoded RGBA feature score",
            "reference": "real Sprites validation",
            "comparable": "no",
            "metrics": ROOT / "reports/eval/sprites_patchvq16_decoded/metrics.csv",
            "grid": ROOT / "reports/eval/sprites_patchvq16_decoded/temp_1_topk_16_grid.png",
            "decision": "Do not promote",
            "notes": "Learned-token pipeline works, but samples are blockier and metric is separate.",
        },
    ]


def collect_records() -> list[dict]:
    records = []
    for branch in branches():
        row = best_row(branch["metrics"])
        record = {
            "rank": branch["rank"],
            "branch": branch["branch"],
            "run": branch["run"],
            "training_data": branch["training_data"],
            "metric_family": branch["metric_family"],
            "reference": branch["reference"],
            "directly_comparable_to_main": branch["comparable"],
            "best_score": f"{float(row['feature_fid']):.5f}",
            "temperature": row["temperature"],
            "top_k": fmt_top_k(row.get("top_k", "")),
            "refinement_steps": branch.get("refinement_steps", ""),
            "opaque_ratio": row.get("opaque_ratio", ""),
            "edge_density": row.get("edge_density", ""),
            "decision": branch["decision"],
            "notes": branch["notes"],
            "metrics_path": str(branch["metrics"].relative_to(ROOT)).replace("\\", "/"),
            "sample_grid": str(branch["grid"].relative_to(ROOT)).replace("\\", "/"),
        }
        records.append(record)
    return records


def write_decision_tables(records: list[dict]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FINAL_DIR / "model_decision_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Final Model Decision Table",
        "",
        "Lower feature scores are better. Rows marked comparable use the shared real Sprites validation palette-token evaluator. Patch-VQ uses a decoded RGBA evaluator, and generated-keep uses its own generated validation reference, so those scores are not direct winner comparisons.",
        "",
        "| Rank | Branch | Best score | Setting | Comparable | Decision |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ]
    for record in records:
        setting = f"temp={record['temperature']}, top_k={record['top_k']}"
        if record["refinement_steps"]:
            setting += f", steps={record['refinement_steps']}"
        lines.append(
            f"| {record['rank']} | {record['branch']} | `{record['best_score']}` | "
            f"{setting} | {record['directly_comparable_to_main']} | {record['decision']} |"
        )

    lines.extend(["", "## Notes", ""])
    for record in records:
        lines.append(
            f"- **{record['branch']}**: {record['notes']} "
            f"Metrics: `{record['metrics_path']}`. Grid: `{record['sample_grid']}`."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "`var_sprites_v0_full` remains the main result. HMAR is the closest ablation on the same real-validation metric, but still does not beat the real-only VAR baseline.",
            "",
        ]
    )
    (FINAL_DIR / "model_decision_table.md").write_text("\n".join(lines), encoding="utf-8")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, base_size: int) -> ImageFont.ImageFont:
    size = base_size
    while size > 10:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
    return load_font(10)


def make_labeled_sheet(src: Path, dst: Path, title: str, subtitle: str) -> None:
    title_font = load_font(34)
    subtitle_font = load_font(22)
    image = Image.open(src).convert("RGBA")
    width, height = image.size
    header_height = 116
    sheet = Image.new("RGBA", (width, height + header_height), (8, 9, 10, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), title, fill=(245, 245, 245, 255), font=title_font)
    draw.text((24, 62), subtitle, fill=(185, 190, 196, 255), font=subtitle_font)
    sheet.alpha_composite(image, (0, header_height))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)


def write_sample_sheets(records: list[dict]) -> None:
    make_labeled_sheet(
        ROOT / "reports/eval/sprites_v0_full/temp_0.8_topk_8_grid.png",
        FINAL_DIR / "final_main_var_sample_sheet.png",
        "Main result: real-only VAR",
        "var_sprites_v0_full | temp=0.8 | top_k=8 | score=0.00147",
    )
    make_labeled_sheet(
        ROOT / "reports/eval/hmar_sprites_refinement_ablation/steps_1/temp_0.8_topk_8_grid.png",
        FINAL_DIR / "final_hmar_sample_sheet.png",
        "Ablation: HMAR masked refinement",
        "hmar_sprites_v0_full | steps=1 | temp=0.8 | top_k=8 | score=0.00189",
    )
    make_labeled_sheet(
        ROOT / "reports/eval/sprites_patchvq16_decoded/temp_1_topk_16_grid.png",
        FINAL_DIR / "final_patchvq_sample_sheet.png",
        "Ablation: decoded patch-VQ VAR",
        "var_sprites_patchvq16_v0_full | temp=1.0 | top_k=16 | decoded score=0.04689",
    )

    title_font = load_font(34)
    label_font = load_font(20)
    small_font = load_font(16)
    thumb_w = 448
    thumb_h = 448
    label_h = 78
    margin = 24
    cols = 2
    rows = 3
    sheet_w = cols * thumb_w + (cols + 1) * margin
    sheet_h = 72 + rows * (label_h + thumb_h) + (rows + 1) * margin
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (8, 9, 10, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), "Final branch comparison", fill=(245, 245, 245, 255), font=title_font)

    for idx, record in enumerate(records):
        col = idx % cols
        row = idx // cols
        x = margin + col * (thumb_w + margin)
        y = 72 + margin + row * (label_h + thumb_h + margin)
        grid = Image.open(ROOT / record["sample_grid"]).convert("RGBA")
        grid = grid.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
        label = f"{record['rank']}. {record['branch']}"
        setting = f"score={record['best_score']} | temp={record['temperature']} | top_k={record['top_k']}"
        if record["refinement_steps"]:
            setting += f" | steps={record['refinement_steps']}"
        draw.text((x, y), label, fill=(245, 245, 245, 255), font=fit_text(draw, label, thumb_w, 22))
        draw.text((x, y + 30), setting, fill=(185, 190, 196, 255), font=fit_text(draw, setting, thumb_w, 16))
        decision_color = (121, 214, 167, 255) if record["rank"] == 1 else (230, 188, 105, 255)
        draw.text((x, y + 52), record["decision"], fill=decision_color, font=small_font)
        sheet.alpha_composite(grid, (x, y + label_h))

    sheet.save(FINAL_DIR / "final_branch_comparison_sheet.png")


def main() -> None:
    records = collect_records()
    write_decision_tables(records)
    write_sample_sheets(records)
    for path in (
        FINAL_DIR / "model_decision_table.csv",
        FINAL_DIR / "model_decision_table.md",
        FINAL_DIR / "final_branch_comparison_sheet.png",
        FINAL_DIR / "final_main_var_sample_sheet.png",
        FINAL_DIR / "final_hmar_sample_sheet.png",
        FINAL_DIR / "final_patchvq_sample_sheet.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
