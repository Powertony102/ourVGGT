import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    return p.parse_args()

def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "average_metrics_report.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_file

def is_numeric(v):
    if v is None:
        return False
    try:
        if isinstance(v, str) and v.strip() == "":
            return False
        float(v)
        return True
    except Exception:
        return False

def find_eval_root():
    candidates = []
    here = Path(__file__).parent
    candidates.append(here)
    st = here / "structure.txt"
    if st.exists():
        return here
    fixed = Path("/Users/lixinze/Library/CloudStorage/OneDrive-个人/Research/20251113_OurVGGT/code/subvggt")
    if fixed.exists():
        return fixed
    return here

def iter_metric_files_from_fs(root: Path):
    names = {"metrics.json"}
    for p in root.rglob("*"):
        if p.is_file():
            if p.name in names:
                yield p
            elif p.suffix.lower() == ".csv":
                yield p
            elif p.suffix.lower() == ".json" and "metrics" in p.name:
                yield p

def iter_metric_files_from_structure(root: Path):
    st = root / "structure.txt"
    if not st.exists():
        return []
    lines = []
    try:
        with st.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logging.error("无法读取 structure.txt %s", str(e))
        return []
    stack = []
    files = []
    for raw in lines:
        line = raw.rstrip("\n")
        if "├──" in line or "└──" in line:
            prefix, name = line.split("──", 1)
            name = name.strip()
            depth = prefix.count("│")
            while len(stack) > depth:
                stack.pop()
            if "." in name:
                candidate = root.joinpath(*stack, name)
                if name.endswith(".json") or name.endswith(".csv"):
                    files.append(candidate)
            else:
                stack.append(name)
    return files

def parse_json_file(p: Path):
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("JSON 解析失败 %s %s", str(p), str(e))
        return []
    rows = []
    if isinstance(data, dict):
        rows.append(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
            else:
                logging.warning("忽略非字典项 %s", str(p))
    else:
        logging.warning("忽略非对象 JSON %s", str(p))
    return rows

def parse_csv_file(p: Path):
    rows = []
    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logging.error("CSV 解析失败 %s %s", str(p), str(e))
    return rows

def normalize_row(row, source):
    out = {"source": str(source)}
    for k, v in row.items():
        if v is None:
            continue
        out[k] = v
    return out

def detect_groups(row):
    g = {}
    for key in ["model", "dataset"]:
        if key in row and str(row[key]).strip() != "":
            g[key] = str(row[key])
    return g

def collect_metrics(rows):
    metric_keys = set()
    for r in rows:
        for k, v in r.items():
            if k in {"source", "model", "dataset", "scene", "split"}:
                continue
            if is_numeric(v):
                metric_keys.add(k)
    return sorted(metric_keys)

def compute_aggregates(rows, metric_keys):
    overall = {k: {"sum": 0.0, "count": 0} for k in metric_keys}
    groups = {}
    for r in rows:
        g = detect_groups(r)
        gkey = tuple(sorted(g.items())) if g else (("all", "all"),)
        if gkey not in groups:
            groups[gkey] = {k: {"sum": 0.0, "count": 0} for k in metric_keys}
        for k in metric_keys:
            v = r.get(k, None)
            if is_numeric(v):
                fv = float(v)
                overall[k]["sum"] += fv
                overall[k]["count"] += 1
                groups[gkey][k]["sum"] += fv
                groups[gkey][k]["count"] += 1
            else:
                if v is not None:
                    logging.warning("非数值字段 已忽略 %s=%s 源 %s", k, str(v), r.get("source", ""))
    return overall, groups

def format_group_key(gkey):
    if gkey == (("all", "all"),):
        return "overall"
    return ", ".join([f"{k}={v}" for k, v in gkey])

def write_report(out_dir: Path, metric_keys, overall, groups):
    report = out_dir / "average_metrics_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("[Overall]\n")
        for k in metric_keys:
            c = overall[k]["count"]
            m = overall[k]["sum"] / c if c > 0 else math.nan
            f.write(f"{k}\t{m:.4f}\t{c}\n")
        for gkey, agg in groups.items():
            if gkey == (("all", "all"),):
                continue
            f.write(f"\n[Group] {format_group_key(gkey)}\n")
            for k in metric_keys:
                c = agg[k]["count"]
                m = agg[k]["sum"] / c if c > 0 else math.nan
                f.write(f"{k}\t{m:.4f}\t{c}\n")
    return report

def write_backup_csv(out_dir: Path, metric_keys, rows):
    backup = out_dir / "raw_metrics_backup.csv"
    header = ["source", "model", "dataset"] + metric_keys
    with backup.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            line = {h: r.get(h, "") for h in header}
            w.writerow(line)
    return backup

def main():
    args = parse_args()
    out_dir = Path(args.path)
    setup_logging(out_dir)
    root = find_eval_root()
    files = list(iter_metric_files_from_structure(root))
    if not files:
        files = list(iter_metric_files_from_fs(root))
    if not files:
        logging.error("未找到评估结果文件")
        return
    rows = []
    for p in files:
        if p.suffix.lower() == ".json":
            parsed = parse_json_file(p)
        elif p.suffix.lower() == ".csv":
            parsed = parse_csv_file(p)
        else:
            parsed = []
        for row in parsed:
            norm = normalize_row(row, p)
            rows.append(norm)
    if not rows:
        logging.error("没有可解析的评估数据")
        return
    metric_keys = collect_metrics(rows)
    if not metric_keys:
        logging.error("未检测到数值型指标")
        return
    overall, groups = compute_aggregates(rows, metric_keys)
    report = write_report(out_dir, metric_keys, overall, groups)
    backup = write_backup_csv(out_dir, metric_keys, rows)
    logging.info("报告生成 %s", str(report))
    logging.info("原始数据备份生成 %s", str(backup))

if __name__ == "__main__":
    main()
