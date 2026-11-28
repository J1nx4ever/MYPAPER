"""Index original evidence and snapshot readable BG128 source code and short results.

Run from the repository with its anylogic-env Python. The default verifies all
previously copied files. --collect creates or verifies the independent copies.
Historical run trees, per-run frozen packages and model/data files stay in place.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

AUDIT = Path(__file__).resolve().parent
PAPER = AUDIT.parent
ROOT = PAPER.parent
PROJ = ROOT / "experiments/lagrangian_ppo_v1"
THEME = Path("token_attention_001/business_graph_summary_009")
PLANS = PROJ / "change_plans" / THEME
REPORTS = PROJ / "reports" / THEME
TRAIN, ABLATION, COMPARE = "01_训练与选模", "02_消融实验", "03_对比实验"
ROBUST, MECHANISM, COMMON = "04_泛化与鲁棒性", "05_机制与仿真核验", "06_共享代码"
CATEGORIES = (TRAIN, ABLATION, COMPARE, ROBUST, MECHANISM, COMMON)
MAP_FILE = AUDIT / "file_map.tsv.gz"
REFS = AUDIT / "原位置索引.tsv"
NOV24_ARCHIVE = REPORTS / "paper_review_20261124/stage1_evidence_001/paper_snapshot_20261124.zip"
CODE_SUFFIX = {".py", ".java", ".proto", ".sh"}
CODE_MAX = 512 * 1024
SUMMARY_MAX = 192 * 1024
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "target",
             "node_modules", "stage1_evidence_001", "configs", "run_configs",
             "java_claims", "material", "rollouts", "archives"}
FINAL_REPORTS = {
    "final_ablation_comparison_20261123.json", "paired_evaluation_summary.json",
    "settlement_comparison.json", "u32_metrics_review.json", "checkpoint_selection.json",
    "seed_scope_audit.json", "training_curves_32_updates.json", "three_seed_learning_evidence.json",
    "same_host_compute_probe.json", "dump_to_load_full_seed_frozen_replay_20261124.json",
    "seed_plateau_dispatch_replay_20261124.json", "completion_validation.json",
}


def walker(root):
    for folder, children, files in os.walk(root):
        children[:] = sorted(n for n in children if n not in SKIP_DIRS
                             and not n.startswith(("prepared_", "archive_", "minibatch_replay_"))
                             and not (Path(folder) / n).is_symlink())
        for name in sorted(files):
            p = Path(folder) / name
            if not p.is_symlink() and not name.startswith(".") and p.is_file():
                yield p


def topic_group(name):
    if name in {"structural_ablation_001", "gat_weak_ablation_001"}:
        return ABLATION
    if name in {"baseline_comparison_001", "reinforce_comparison_001"}:
        return COMPARE
    if name.startswith(("generalization_", "fleet_outage", "variable_topology_")):
        return ROBUST
    if name in {"paper_review_20261128", "formal_training_001", "training_extension_001", "optimization_stability_001",
                "checkpoint_evaluation_001", "trained_analysis_001", "netron_checkpoint13_001"}:
        return TRAIN
    return MECHANISM


def run_group(name):
    if name.startswith(("bg128-baseline-", "bg128-dispatch-", "bg128-or-", "bg128-gat1-s01-u32-grade-", "bg128-reinforce-")):
        return COMPARE
    if name.startswith(("bg128-ablation-", "bg128-gat-weak-", "bg128-gat1-replication-", "bg128-gat-depth-")):
        return ABLATION
    if name.startswith(("bg128-fleet-", "bg128-full-s02-u32-fleet", "bg128-u32-generalization-", "site-layout-")):
        return ROBUST
    if name.startswith(("bg128-first-window-", "bg128-u13-discovery", "bg128-synthetic-", "west-pit-", "topology-simulation-")):
        return MECHANISM
    return TRAIN


def is_code(path):
    return path.suffix in CODE_SUFFIX or path.name.endswith((".java.txt", ".alp.txt", ".java.inc"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def classify(evidence_files):
    # Keep old selections stable even if they no longer occur in the active catalogue.
    copies, refs = {}, {}
    if MAP_FILE.is_file():
        with gzip.open(MAP_FILE, "rt", encoding="utf-8", newline="") as stream:
            copies.update((ROOT / row["source"], PAPER / row["destination"])
                          for row in csv.DictReader(stream, delimiter="\t"))
    if REFS.is_file():
        with REFS.open(encoding="utf-8", newline="") as stream:
            refs.update((ROOT / row["source"], (row["category"], row["reason"]))
                        for row in csv.DictReader(stream, delimiter="\t"))

    def add(src, group, kind, rel):
        dest = PAPER / group / kind / rel
        if src in copies and copies[src] != dest:
            raise ValueError(f"conflicting classification for {src}")
        copies[src] = dest

    def reference(src, group, reason):
        refs[src] = (group, reason)

    for root, kind in ((PLANS, "方法与实现"), (REPORTS, "封存摘要")):
        for src in walker(root):
            rel = src.relative_to(root)
            group = topic_group(rel.parts[0])
            size = src.stat().st_size
            name = src.name
            if root == PLANS and rel.parts[0] == "variable_topology_training_001" and name in {
                "launch_identity.json", "run_index_seed_manifest.csv"
            }:
                panel = {"paired_50_001": "A/B 两路网共 100 班",
                         "b_finetune_001": "B 图训练、验证及测试共 258 班",
                         "paired_severity_001": "A 图停工强度与策略比较共 270 班"}
                scope = panel.get(src.parent.name, "本活动")
                reference(src, group, f"{scope}的逐班目录与 run/seed 身份入口，可按清单定位各班。")
            if is_code(src) and size <= CODE_MAX:
                add(src, group, "结果分析代码" if root == REPORTS else kind, rel)
            elif src.suffix == ".md" and size <= SUMMARY_MAX:
                add(src, group, kind, rel)
            elif (root == PLANS and rel == Path("schema.json")):
                add(src, group, kind, rel)
            elif (root == REPORTS and size <= SUMMARY_MAX and
                  (name in FINAL_REPORTS or (name == "evaluation.json" and rel.parts[0] == "fleet_outage_resilience_001"))):
                add(src, group, kind, rel)
            elif root == REPORTS and src.suffix == ".json" and any(
                word in name for word in ("evaluation", "comparison", "summary", "complete", "validation", "result", "audit", "preparation")
            ):
                reference(src, group, "原封存结果、核验或准备记录；保留原位置，可从总表或源报告定位。")
        if root == PLANS:
            # Record package roots, never enumerate their run-level contents.
            for current, dirs, _ in os.walk(root):
                for name in dirs:
                    if name.startswith("prepared_"):
                        src = Path(current) / name
                        reference(src, topic_group(src.relative_to(root).parts[0]), "冻结启动包；配置、清单与 Java 身份按原位置保存。")
                dirs[:] = [n for n in dirs if n not in SKIP_DIRS and not n.startswith(("prepared_", "archive_"))]

    for src in (PROJ / "runs").iterdir():
        if src.name.startswith(("business-graph-", "bg128-", "west-pit-", "topology-simulation-")) and src.is_dir():
            group = run_group(src.name)
            label = "原始物料目录（未递归读取）。" if src.name.endswith("_materials") else "原始运行目录（未递归读取）；完成、开发、恢复或中断状态以证据总表为准。"
            reference(src, group, label)

    for src in walker(ROOT / "summary/interpretability_eval"):
        rel = src.relative_to(ROOT / "summary")
        if (is_code(src) or src.name == "README.md") and src.stat().st_size <= CODE_MAX:
            add(src, MECHANISM, "可解释性分析代码", rel)
        elif src.suffix in {".pdf", ".png"}:
            reference(src, MECHANISM, "已有解释性图的原件。")
    archived_interpret = PROJ / "reports/interpretability_eval"
    for src in walker(archived_interpret):
        if src.suffix == ".md" and src.stat().st_size <= SUMMARY_MAX:
            add(src, MECHANISM, "封存摘要", src.relative_to(PROJ / "reports"))
        elif src.suffix == ".json":
            reference(src, MECHANISM, "解释性统计、轨迹重放或图源数据。")
    off = PROJ / "reports/fleet_outage_resilience_001/anylogic_offline.json"
    if off.exists():
        reference(off, ROBUST, "车队活动的离线 AnyLogic 核验。")

    for src in walker(ROOT / "policy4"):
        if is_code(src) or (src.suffix == ".md" and src.stat().st_size <= SUMMARY_MAX):
            add(src, COMPARE, "传统方案代码", src.relative_to(ROOT / "policy4"))
    for name in ("python", "proto-src"):
        for src in walker(ROOT / name):
            if is_code(src) or src.name == "pom.xml":
                add(src, COMMON, "仓库公共实现", src.relative_to(ROOT))

    # Copy entire BG128 and baseline implementations. For their historical
    # dependencies select the imported modules instead of entire legacy campaigns.
    packages = {p.name:p for p in PROJ.iterdir() if p.is_dir() and (p/"__init__.py").is_file()}
    for name, folder in packages.items():
        if name.startswith(("business_graph_", "traditional_dispatch_")):
            for src in walker(folder):
                if is_code(src):
                    if name in {"business_graph_reinforce_v1", "business_graph_matched_reinforce_v1"}:
                        add(src, COMPARE, "算法实现", src.relative_to(PROJ))
                    else:
                        add(src, COMMON, "实验共用模块", src.relative_to(PROJ))
    todo = [p for p in copies if p.suffix == ".py"]
    scanned = set()
    while todo:
        src = todo.pop()
        if src in scanned:
            continue
        scanned.add(src)
        for n in ast.walk(ast.parse(src.read_bytes(), filename=str(src))):
            modules = []
            if isinstance(n, ast.Import):
                modules.extend(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                prefix = n.module or ""
                if n.level:
                    try:
                        pkg = src.parent.relative_to(PROJ).parts
                    except ValueError:
                        continue
                    if not pkg or pkg[0] not in packages or n.level > len(pkg):
                        continue
                    pkg = pkg[:len(pkg) - n.level + 1]
                    prefix = ".".join((*pkg, prefix)) if prefix else ".".join(pkg)
                modules.extend([prefix, *(prefix + "." + a.name for a in n.names)])
            for module in modules:
                parts = module.split(".")
                if parts[0] not in packages:
                    continue
                path = PROJ.joinpath(*parts)
                module_file = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
                if module_file.is_file() and module_file not in copies:
                    add(module_file, COMMON, "实验共用模块", module_file.relative_to(PROJ))
                    todo.append(module_file)
                init = packages[parts[0]] / "__init__.py"
                if init not in copies:
                    add(init, COMMON, "实验共用模块", init.relative_to(PROJ))
                    todo.append(init)
    for label in ("proto", "proto-src"):
        for src in walker(PROJ / label):
            if is_code(src) or src.name=="pom.xml":
                add(src, COMMON, "实验共用模块", src.relative_to(PROJ))
    for src in (PROJ/"storage_requirements.txt", PROJ/"ENVIRONMENT_ACCEPTANCE_001.md"):
        add(src, COMMON, "实验共用模块", src.relative_to(PROJ))
    for label in ("anylogic/DRL-PPO.alp", "anylogic/3d/DRL-PPO-baseline1/DRL-PPO-baseline1.alp"):
        src=PROJ / label
        if src.is_file():
            reference(src, COMMON, "AnyLogic 模型原件；未读取、复制或编辑。")
    progress = REPORTS / "paper_review_20261128/training_progress.json"
    if progress.is_file() and progress.stat().st_size <= 256 * 1024:
        add(progress, TRAIN, "封存摘要", progress.relative_to(REPORTS))
    # The paper catalogue selects the final evidence. Inspect only its named
    # small summaries and top-level suite seals, never walk the underlying runs.
    catalog = PAPER / "00_证据总表/evidence_catalog.json"
    if catalog.is_file():
        for item in json.loads(catalog.read_bytes())["sources"]:
            src = ROOT / item["source"]
            if src in copies or not src.is_file():
                continue
            if src.is_relative_to(REPORTS) and src.suffix == ".json":
                rel = src.relative_to(REPORTS)
                if ("stage1_evidence_001" not in rel.parts and src.stat().st_size <= 256 * 1024
                        and "preflight" not in src.name and "preparation" not in src.name):
                    add(src, topic_group(rel.parts[0]), "封存摘要", rel)
            elif src.is_relative_to(PROJ / "runs") and src.parent.parent == PROJ / "runs" and src.name == "suite_complete.json" and src.stat().st_size <= 64 * 1024:
                add(src, run_group(src.parent.name), "活动完成回执", src.relative_to(PROJ / "runs"))
            if src not in copies and src not in refs:
                if src.is_relative_to(PROJ / "runs"):
                    group = run_group(src.relative_to(PROJ / "runs").parts[0])
                elif src.is_relative_to(REPORTS):
                    group = topic_group(src.relative_to(REPORTS).parts[0])
                elif src.is_relative_to(PLANS):
                    group = topic_group(src.relative_to(PLANS).parts[0])
                else:
                    group = MECHANISM if "interpretability_eval" in src.parts else COMMON
                reference(src, group, "阶段一总表直接引用的原件；记录定位，内容保留在源目录。")
    # An update may supply prospective evidence before the new workbook is published.
    # Subsequent invocations retain these selections from the installed maps above.
    selected = set()
    for file in evidence_files:
        data = json.loads(file.read_text(encoding="utf-8"))
        for row in data.get("records", []):
            selected.update(row.get("sources", []))
            selected.update(row.get("code", []))
        for key in ("code", "light_results", "copy_extra", "diagnostic_source_files", "large_originals"):
            selected.update(data.get(key, []))
        selected.update(row["source"] for row in data.get("groups", []) if row.get("source"))
        selected.update(data.get("notes", {}).get("archive", {}).get("sources", [])
                        if isinstance(data.get("notes"), dict) else [])
    for label in sorted(selected):
        src = ROOT / label
        if not src.is_relative_to(ROOT) or ".." in src.parts or src.is_symlink() or not src.is_file():
            raise ValueError(f"missing or invalid evidence path {src}")
        if src in copies:
            continue
        size = src.stat().st_size
        if src.is_relative_to(PLANS):
            rel, kind = src.relative_to(PLANS), "方法与实现"
            group = topic_group(rel.parts[0])
        elif src.is_relative_to(REPORTS):
            rel, kind = src.relative_to(REPORTS), "封存摘要"
            group = topic_group(rel.parts[0])
        elif src.is_relative_to(PROJ / "runs"):
            rel = src.relative_to(PROJ / "runs")
            group = run_group(rel.parts[0])
            kind = "活动完成回执" if src.name == "suite_complete.json" else "活动记录"
        else:
            reference(src, COMMON, "历史归档与来源核验入口。")
            continue
        if is_code(src) and size <= CODE_MAX:
            add(src, group, kind, rel)
        elif (size <= 256 * 1024 and
              src.suffix in {".json", ".csv", ".md", ".txt", ".xlsx"} and
              (not src.is_relative_to(PROJ / "runs") or src.name in {
                  "suite_complete.json", "run_status.json", "launch_intent.json",
                  "training_progress.json", "verification.json"
              })):
            add(src, group, kind, rel)
        else:
            reference(src, group, "总表对应原件（含逐班明细、逐状态输出或活动账本）；按路径查阅。")
    if len(copies) != len(set(copies.values())):
        raise ValueError("multiple source paths map to one copy")
    return copies, refs


def save_map(entries):
    fd, tmp = tempfile.mkstemp(dir=AUDIT, prefix=".file_map.")
    os.close(fd)
    try:
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
             io.TextIOWrapper(packed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("source","destination","bytes","sha256"))
            writer.writerows(entries)
        os.replace(tmp, MAP_FILE)
    finally:
        Path(tmp).unlink(missing_ok=True)


def archived_previous_version(src, dest):
    if src not in {PLANS / "README.md", PLANS / "fleet_outage_resilience_001/README.md"}:
        return False
    member = str(dest.relative_to(ROOT))
    with ZipFile(NOV24_ARCHIVE) as archive:
        with gzip.open(io.BytesIO(archive.read("paper/07_归档核验/file_map.tsv.gz")), "rt",
                       encoding="utf-8", newline="") as old:
            entry = next((row for row in csv.DictReader(old, delimiter="\t")
                          if row["source"] == str(src.relative_to(ROOT))), None)
        return (entry is not None and entry["destination"] == str(dest.relative_to(PAPER)) and
                hashlib.sha256(archive.read(member)).hexdigest() == entry["sha256"] == digest(dest))


def collect(evidence_files):
    copies, refs = classify(evidence_files)
    entries=[]
    for src,dest in sorted(copies.items()):
        dest.parent.mkdir(parents=True,exist_ok=True)
        expected=digest(src)
        if dest.is_symlink():
            raise ValueError(f"Refusing symlink destination {dest}")
        if not src.is_file() or src.is_symlink():
            raise ValueError(f"missing or invalid source {src}")
        if not dest.exists() or (digest(dest) != expected and archived_previous_version(src, dest)):
            fd, name=tempfile.mkstemp(dir=dest.parent,prefix=".archive.")
            os.close(fd)
            try:
                shutil.copy2(src,name)
                os.replace(name,dest)
            finally:
                Path(name).unlink(missing_ok=True)
        if src.samefile(dest) or digest(dest)!=expected or dest.stat().st_size!=src.stat().st_size:
            raise ValueError(f"Copy differs from source or is not independent: {dest}")
        entries.append((src.relative_to(ROOT),dest.relative_to(PAPER),src.stat().st_size,expected))
    save_map(entries)
    with REFS.open("w",encoding="utf-8",newline="") as out:
        writer=csv.writer(out,delimiter="\t",lineterminator="\n")
        writer.writerow(("category","source","reason"))
        writer.writerows((group,src.relative_to(ROOT),reason) for src,(group,reason) in sorted(refs.items()) if src not in copies)
    return len(entries),sum(x[2] for x in entries)


def verify():
    total=size=0
    with gzip.open(MAP_FILE,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            src,dest=ROOT/row["source"],PAPER/row["destination"]
            if dest.is_symlink() or not dest.is_file() or dest.stat().st_size!=int(row["bytes"]) or digest(dest)!=row["sha256"]:
                raise ValueError(f"invalid paper copy {dest}")
            if not src.is_file() or src.is_symlink() or src.samefile(dest) or digest(src)!=row["sha256"]:
                raise ValueError(f"source changed since snapshot or shares inode: {src}")
            total+=1
            size+=int(row["bytes"])
    return total,size


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect",action="store_true",help="copy new eligible files and rebuild the small-copy manifest")
    parser.add_argument("--evidence-input", type=Path, action="append", default=[],
                        help="include a validated evidence update's selected lightweight results and code")
    args=parser.parse_args()
    count,size=collect(args.evidence_input) if args.collect else verify()
    print(f"Verified {count} independent file copies ({size:,} bytes)")
