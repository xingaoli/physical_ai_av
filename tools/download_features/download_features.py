#!/usr/bin/env python3
# ============================================================================
# 通用 chunk 特征下载器（physical_ai_av / PhysicalAI-Autonomous-Vehicles）
#
# 下载任意 feature 的全部 chunk 到本地目录，例如：
#   egomotion  -> labels/egomotion/egomotion.chunk_XXXX.zip      (~35MB/chunk)
#   camera_front_wide_120fov -> camera/.../*.zip                 (GB 级/chunk)
#   camera_intrinsics -> calibration/.../*.parquet               (KB 级/chunk)
# feature 名单见 <local_dir>/features.csv（本脚本会给出提示）。
#
# 结构（针对 hf-mirror 的三个坑设计）：
#   1. dataset.py 的 download_chunk_features 没有 return，单文件下载失败不可能
#      抛到调用方 -> 唯一可信校验是磁盘上文件落地 + 完整性抽查
#      （zip 读中央目录 / parquet 查 PAR1 魔数）
#   2. 镜像多连接会挂死(不抛异常、字节不流动) -> 下载放在子进程里，
#      看门狗监控 *.incomplete 字节增长：停滞 --stall-secs 秒子进程自杀(exit 3)，
#      父进程空闲兜底 kill 整个进程组；每击杀一次并发自动 -1（最低 1）
#   3. 断点续传(Range resume)在镜像上会触发 416 -> 启动时清掉全部
#      .incomplete/.lock，永远整文件下载；已完成的文件不受影响
#
# 模式：
#   parent（默认）：编排 —— 批次调度/子进程监控/校验/重试/磁盘守卫，自己不碰网络
#   child        ：--mode child --chunks 0-7  实际下载一个批次
#   list         ：--mode list 从 hub 拉权威 chunk 列表写到 json
#
# 一般不直接跑本文件，用 start_download.sh（后台/脱离终端/崩溃自动重启）。
# 停止：bash start_download.sh stop --feature <f>，随时重启自动续传（按磁盘跳过）。
# ============================================================================
import os

# 必须在 import huggingface_hub 之前设置（child 模式导入 hf 时生效）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 镜像站
os.environ["HF_HUB_DISABLE_XET"] = "1"          # 避免 xet (cas-bridge) 超时
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 日志走父进程前缀行，关掉 tqdm 噪声

import argparse
import csv
import json
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

PY = sys.executable
DEFAULT_LOCAL_DIR = "/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-all"
SCRIPT_DIR = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def list_file(feature: str) -> Path:
    return SCRIPT_DIR / f".chunk_list_{feature}.json"


def parse_chunks(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def file_ok(path: Path) -> bool:
    """完整性抽查：zip 读中央目录（能发现截断）；parquet 查首尾 PAR1 魔数；其余查非空。"""
    try:
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                return len(zf.namelist()) > 0
        if path.suffix == ".parquet":
            with open(path, "rb") as f:
                if f.read(4) != b"PAR1":
                    return False
                f.seek(-4, 2)
                return f.read(4) == b"PAR1"
        return path.stat().st_size > 0
    except Exception:
        return False


def feature_template_from_csv(local_dir: Path, feature: str) -> tuple[str, str]:
    """从 <local_dir>/features.csv 解析 feature 的 chunk_path 模板。
    返回 (相对目录, 文件名模板)，如 ('labels/egomotion', 'egomotion.chunk_{chunk_id:04d}.zip')。
    feature 不存在时列出全部可选 feature 并报错。"""
    csv_path = local_dir / "features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} 不存在；先跑一次 --mode list 让其下载")
    with open(csv_path, newline="") as f:
        rows = {r["feature"]: r["chunk_path"] for r in csv.DictReader(f)}
    if feature not in rows:
        avail = "\n".join(f"  - {k}" for k in sorted(rows))
        raise ValueError(f"未知 feature '{feature}'，features.csv 中可选：\n{avail}")
    template = rows[feature]
    rel_dir, fname = template.rsplit("/", 1)
    return rel_dir, fname


# ---------------------------------------------------------------- child ----
def init_interface(local_dir: str, retries: int = 5):
    from physical_ai_av.dataset import PhysicalAIAVDatasetInterface

    last = None
    for attempt in range(1, retries + 1):
        try:
            return PhysicalAIAVDatasetInterface(
                token=True,
                local_dir=local_dir,
                confirm_download_threshold_gb=float("inf"),
            )
        except Exception as e:
            last = e
            log(f"[child] 接口初始化失败({attempt}/{retries}): {type(e).__name__}: {e}")
            time.sleep(60)
    raise RuntimeError(f"接口初始化连续失败: {last}")


class Watchdog(threading.Thread):
    """监控 feature 下载目录 *.incomplete 字节增长：停滞过久自杀；每 60s 打心跳行。"""

    def __init__(self, local_dir: str, rel_dir: str, stall_secs: int):
        super().__init__(daemon=True)
        self.dir = Path(local_dir) / ".cache" / "huggingface" / "download" / rel_dir
        self.stall_secs = stall_secs
        self.last_sizes: dict[str, int] = {}
        self.last_change = time.time()
        self.last_hb = 0.0

    def run(self):
        while True:
            time.sleep(30)
            sizes = (
                {p.name: p.stat().st_size for p in self.dir.glob("*.incomplete")}
                if self.dir.is_dir() else {}
            )
            now = time.time()
            if sizes and sizes != self.last_sizes:
                self.last_change = now
            self.last_sizes = sizes
            if now - self.last_hb >= 60:
                if sizes:
                    log(f"[hb] 下载中 {len(sizes)} 文件, {sum(sizes.values()) / 1e6:.1f}MB")
                else:
                    log("[hb] 无进行中的下载文件")
                self.last_hb = now
            if sizes and now - self.last_change > self.stall_secs:
                log(f"[wd] ⚠️ {self.stall_secs}s 无字节进展（镜像挂死），子进程自杀")
                os._exit(3)


def run_child(args) -> int:
    chunks = parse_chunks(args.chunks)
    log(f"[child] 批次开始: {len(chunks)} 个 chunk, workers={args.workers}")
    ds = init_interface(args.local_dir)

    template = ds.features.features_df.at[args.feature, "chunk_path"]
    rel_dir, fname_t = template.rsplit("/", 1)
    local_dir = Path(args.local_dir)
    egress = local_dir / rel_dir

    # presence 语义与 dataset.py 一致：feature 不在表里 -> 视为所有 chunk 都有
    presence = ds.chunk_feature_presence
    if args.feature in presence.columns:
        todo = [c for c in chunks if bool(presence.at[c, args.feature])]
    else:
        todo = list(chunks)
    if len(todo) != len(chunks):
        log(f"[child] 跳过 {len(chunks) - len(todo)} 个 hub 上无此 feature 的 chunk")

    Watchdog(args.local_dir, rel_dir, args.stall_secs).start()
    # 注意：download_chunk_features 无返回值，单文件失败不会抛异常，
    # 落地校验在下方及父进程做
    ds.download_chunk_features(todo, features=[args.feature], max_workers=args.workers)

    ok, bad = 0, 0
    for c in todo:
        p = egress / fname_t.format(chunk_id=c)
        if p.exists() and file_ok(p):
            ok += 1
            log(f"[child] ✅ {p.name}")
        else:
            bad += 1
            log(f"[child] ❌ {p.name} 未落地或损坏")
    log(f"[child] 批次结束: 成功 {ok}, 失败 {bad}")
    return 0 if bad == 0 and ok == len(todo) else 1


def run_list(args) -> int:
    ds = init_interface(args.local_dir)
    presence = ds.chunk_feature_presence
    if args.feature in presence.columns:
        chunks = sorted(c for c in presence.index if bool(presence.at[c, args.feature]))
    else:
        chunks = sorted(presence.index)
    list_file(args.feature).write_text(
        json.dumps({"generated": datetime.now().isoformat(), "chunks": chunks})
    )
    log(f"[list] feature '{args.feature}' 权威 chunk 列表: {len(chunks)} 个 "
        f"-> {list_file(args.feature)}")
    return 0


# --------------------------------------------------------------- parent ----
STOP = False
CHILD_PROC: subprocess.Popen | None = None


def _on_signal(signum, frame):
    global STOP
    STOP = True
    log(f"父进程收到信号 {signum}，终止当前子进程并退出...")
    if CHILD_PROC and CHILD_PROC.poll() is None:
        try:
            os.killpg(os.getpgid(CHILD_PROC.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def run_monitored(cmd: list[str], idle_timeout: float, budget: float) -> int | None:
    """跑子进程，转发其输出；空闲/超预算则 kill 整个进程组。返回 rc（None=被 STOP 打断）。"""
    global CHILD_PROC
    log(f"  ▶ 启动子进程: {' '.join(cmd[1:])}")
    CHILD_PROC = p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    q: queue.Queue = queue.Queue()

    def reader():
        assert p.stdout is not None
        for line in p.stdout:
            q.put((time.time(), line.rstrip("\n")))
        q.put((time.time(), None))

    threading.Thread(target=reader, daemon=True).start()
    t0 = last = time.time()
    killed_reason = None
    while True:
        if STOP:
            _on_signal(signal.SIGTERM, None)
            p.wait()
            return None
        try:
            ts, line = q.get(timeout=15)
        except queue.Empty:
            pass
        else:
            if line is None:
                break
            log(f"  │ {line}")
            last = ts
            continue
        now = time.time()
        if now - last > idle_timeout:
            killed_reason = f"空闲 {int(now - last)}s 无输出"
        elif budget and now - t0 > budget:
            killed_reason = f"超过预算 {int(now - t0)}s"
        if killed_reason:
            log(f"  ⚠️ {killed_reason}，kill 子进程组")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
    rc = p.wait()
    if killed_reason and rc == 0:
        rc = -9
    CHILD_PROC = None
    log(f"  ◀ 子进程结束 rc={rc}" + (f"（{killed_reason}）" if killed_reason else ""))
    return rc


def clean_downloads(local_dir: Path, rel_dir: str) -> None:
    """清该 feature 下载目录的 .incomplete/.lock 残留（metadata 不动，完成的文件不受影响）。"""
    d = local_dir / ".cache" / "huggingface" / "download" / rel_dir
    if not d.is_dir():
        return
    n = 0
    for p in d.iterdir():
        if p.name.endswith((".incomplete", ".lock")):
            p.unlink(missing_ok=True)
            n += 1
    if n:
        log(f"🧹 清理 {n} 个缓存残留(incomplete/lock)")


def get_chunk_list(args, local_dir: Path) -> tuple[str, str, list[int]]:
    """返回 (rel_dir, 文件名模板, chunk 列表)。features.csv 缺失时先用 list child 拉齐。"""
    if not (local_dir / "features.csv").exists():
        log("features.csv 不在本地，先跑 list 模式拉取元数据...")
        cmd = [PY, __file__, "--mode", "list", "--feature", args.feature,
               "--local-dir", str(local_dir)]
        for i in range(3):
            if run_monitored(cmd, idle_timeout=900, budget=0) == 0:
                break
            log(f"list 模式失败，第 {i + 2} 次尝试...")
            time.sleep(30)
        else:
            raise RuntimeError("无法拉取元数据(features.csv)")

    rel_dir, fname_t = feature_template_from_csv(local_dir, args.feature)

    if args.chunks:
        return rel_dir, fname_t, parse_chunks(args.chunks)

    lf = list_file(args.feature)
    if lf.exists():
        try:
            chunks = json.loads(lf.read_text())["chunks"]
            if chunks:
                log(f"使用已有 chunk 列表: {len(chunks)} 个（{lf}）")
                return rel_dir, fname_t, chunks
        except Exception:
            pass
    cmd = [PY, __file__, "--mode", "list", "--feature", args.feature,
           "--local-dir", str(local_dir)]
    for i in range(3):
        rc = run_monitored(cmd, idle_timeout=900, budget=0)
        if rc == 0 and lf.exists():
            return rel_dir, fname_t, json.loads(lf.read_text())["chunks"]
        log(f"list 模式失败(rc={rc})，第 {i + 2} 次尝试...")
        time.sleep(30)
    raise RuntimeError("无法获取 chunk 列表")


def run_parent(args) -> int:
    local_dir = Path(args.local_dir)
    rel_dir, fname_t, all_chunks = get_chunk_list(args, local_dir)
    egress = local_dir / rel_dir
    egress.mkdir(parents=True, exist_ok=True)
    chunk_path = lambda c: egress / fname_t.format(chunk_id=c)

    total = len(all_chunks)
    log("=" * 60)
    log(f"feature '{args.feature}' 全量下载: {total} 个 chunk -> {egress}")
    log(f"endpoint={os.environ.get('HF_ENDPOINT')}  起始并发={args.workers}")

    # 镜像的 Range 续传不可靠(416)：启动时清掉半成品，全部整文件下载。
    # 已完成文件不受影响（hf_hub_download 见文件在即跳过）。
    clean_downloads(local_dir, rel_dir)

    def missing_chunks() -> list[int]:
        return [c for c in all_chunks if not chunk_path(c).exists()]

    fail_count: dict[int, int] = {}
    t0 = time.time()
    done0 = total - len(missing_chunks())
    workers = args.workers
    sweep = 0

    while sweep < args.max_sweeps:
        if STOP:
            return 99
        missing = missing_chunks()
        if not missing:
            break
        sweep += 1
        log(f"━━━ 第 {sweep}/{args.max_sweeps} 轮: 待下载 {len(missing)} 个, 并发 {workers} ━━━")

        for i in range(0, len(missing), args.batch_size):
            if STOP:
                return 99
            batch = missing[i : i + args.batch_size]

            while not STOP:  # 磁盘空间守卫
                free_gb = shutil.disk_usage(str(local_dir)).free / 1024**3
                if free_gb >= args.min_free_gb:
                    break
                log(f"⚠️ 磁盘剩余 {free_gb:.1f}GB < {args.min_free_gb}GB，暂停 5 分钟")
                time.sleep(300)

            csv_spec = ",".join(str(c) for c in batch)
            budget = len(batch) * args.per_chunk_budget + 300 if args.per_chunk_budget else 0
            cmd = [PY, __file__, "--mode", "child", "--chunks", csv_spec,
                   "--local-dir", str(local_dir), "--workers", str(workers),
                   "--stall-secs", str(args.stall_secs), "--feature", args.feature]
            rc = run_monitored(cmd, idle_timeout=args.stall_secs + 300, budget=budget)

            # 落地校验（唯一可信信号）
            still_missing = []
            for c in batch:
                p = chunk_path(c)
                if p.exists():
                    if not file_ok(p):
                        log(f"  ❌ {p.name} 完整性校验失败，删除重下")
                        p.unlink(missing_ok=True)
                        still_missing.append(c)
                else:
                    still_missing.append(c)
            for c in still_missing:
                fail_count[c] = fail_count.get(c, 0) + 1

            done = sum(1 for c in all_chunks if chunk_path(c).exists())
            rate = (done - done0) / max(time.time() - t0, 1) * 60
            eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
            log(f"📊 进度 {done}/{total} ({done / total * 100:.1f}%)  "
                f"速率 {rate:.1f} chunk/min  预计剩余 {eta_min:.0f} 分钟"
                + (f"  [本批未完成: {still_missing}]" if still_missing else ""))

            if rc in (3, -9) and workers > 1:  # 挂死被杀 -> 降并发
                workers -= 1
                log(f"⬇️ 检测到镜像挂死，并发降为 {workers}")
            clean_downloads(local_dir, rel_dir)

        missing = missing_chunks()
        if missing and not STOP:
            worst = sorted(missing, key=lambda c: -fail_count.get(c, 0))[:10]
            log(f"本轮剩余 {len(missing)} 个未完成（重试最多: {worst}），{args.sweep_wait}s 后下一轮")
            time.sleep(args.sweep_wait)

    if STOP:
        return 99
    missing = missing_chunks()
    n_files = sum(1 for c in all_chunks if chunk_path(c).exists())
    if missing:
        log(f"❌ 达到最大轮数，仍缺 {len(missing)} 个: {missing[:50]}...")
        return 1
    log(f"🎉 全部完成: {n_files}/{total} 个文件已就绪 -> {egress}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="通用 chunk 特征下载器")
    parser.add_argument("--mode", choices=["parent", "child", "list"], default="parent")
    parser.add_argument("--feature", required=True,
                        help="feature 名，见 <local_dir>/features.csv，如 egomotion")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--chunks", type=str, default=None, help="如 '0-9,20'；缺省=全量(hub 列表)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="每批并发连接数；挂死自动降级")
    parser.add_argument("--stall-secs", type=int, default=600, help="字节无增长视为挂死")
    parser.add_argument("--per-chunk-budget", type=int, default=0,
                        help="每 chunk 秒数预算，0=不限（大文件 feature 保持 0，靠挂死看门狗）")
    parser.add_argument("--max-sweeps", type=int, default=500)
    parser.add_argument("--sweep-wait", type=int, default=60)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    if args.mode == "child":
        return run_child(args)
    if args.mode == "list":
        return run_list(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
