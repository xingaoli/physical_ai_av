# download_features — 通用 chunk 特征下载器

对 `nvidia/PhysicalAI-Autonomous-Vehicles`（经 hf-mirror.com 镜像）**按 feature 全量下载 chunk 文件**的健壮工具：后台运行、脱离终端、断点续传、自动重试、镜像挂死自动检测与降并发。

可下载任意 feature（名单见 `<local_dir>/features.csv`），常用：

| feature | 落地路径（`<local_dir>` 下） | 单 chunk 大小 | 说明 |
|---|---|---|---|
| `egomotion` | `labels/egomotion/egomotion.chunk_XXXX.zip` | ~35MB | 训练必需，~110GB 全量 |
| `egomotion.offline` | `labels/egomotion.offline/*.zip` | ~35MB | 离线标注版 |
| `camera_intrinsics` | `calibration/camera_intrinsics/*.parquet` | KB 级 | 标定，秒下 |
| `sensor_extrinsics` | `calibration/sensor_extrinsics/*.parquet` | KB 级 | 标定 |
| `camera_front_wide_120fov` | `camera/.../*.zip` | GB 级 | 视频，注意磁盘 |
| `lidar_top_360fov` | `lidar/.../*.zip` | GB 级 | 视频，注意磁盘 |

## 快速开始

```bash
cd physical_ai_av/tools/download_features

# 启动（后台、脱离终端，当前终端/harness 退出不受影响）
bash start_download.sh start --feature egomotion

# 查看 / 停止
bash start_download.sh status --feature egomotion
bash start_download.sh stop   --feature egomotion

# 实时日志（按 feature 命名，写在本目录）
tail -f download_egomotion.log
```

下载其他 feature 同理，换 `--feature` 即可；不同 feature 互不干扰（各有独立 pid/日志/chunk 列表，可以同时跑多个）。落地目录默认 `/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-all`，可用 `--local-dir` 覆盖。

## 文件结构

| 文件 | 职责 |
|---|---|
| `download_features.py` | 主体，三种模式：`parent`（编排，默认）/ `child`（子进程下载一批）/ `list`（拉权威 chunk 列表） |
| `run_download.sh` | runner：parent 崩溃 60s 后自动重启（9999 次）；操作者主动停止不重启 |
| `start_download.sh` | 对外入口：`start`/`stop`/`status`，setsid 新会话 + nohup 后台化，pid 防重复启动 |

运行期产物（也在本目录）：`.download_<feature>.pid`、`download_<feature>.log`、`.chunk_list_<feature>.json`（hub 权威 chunk 列表缓存）。

## 设计：为什么不用 `download_chunk_features` 直接循环

对着仓库现有 `tools/download_simple` 的三个坑设计的：

1. **静默失败**：`dataset.py` 的 `download_chunk_features` 没有 `return`，内部 `thread_map` 的单文件异常永远传不到调用方，try/except 抓不到。
   → 本工具以**磁盘落地 + 完整性抽查**为唯一可信信号（zip 读中央目录、parquet 查 PAR1 魔数），失败的 chunk 进入重试。
2. **镜像并行挂死**：hf-mirror 多连接会不抛异常地挂死（连接在、字节不流动）。
   → 下载放在子进程：看门狗线程监控 `*.incomplete` 字节增长，停滞 `--stall-secs`（默认 600s）子进程自杀；父进程另有空闲兜底（无输出 900s）kill 整个进程组。**每击杀一次并发自动 -1**（`--workers` 起步 4，最低 1），镜像抽风时自动退化为串行，恢复后不影响。
3. **Range 续传触发 416**：镜像的断点续传不可靠。
   → 每批结束/任务启动时清掉 `.incomplete/.lock` 残留，永远整文件下载；已完成文件不受影响（hf_hub_download 见文件在即跳过）。

其他保障：按磁盘已有文件跳过（随时 stop/start 续传）、多轮 sweep 重试（500 轮 × 批内重试）、磁盘空间守卫（`--min-free-gb`，默认 20GB，不足暂停 5 分钟轮询）、runner 进程级重启兜底。

## 常用参数（透传给 `download_features.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--feature` | 必填 | feature 名（features.csv 里的 `feature` 列） |
| `--local-dir` | `/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-all` | 数据根目录 |
| `--chunks` | 全量 | 只下指定 chunk，如 `0-99,200`（测试/补漏用） |
| `--batch-size` | 16 | 每批 chunk 数（批是重试/杀进程的粒度） |
| `--workers` | 4 | 每批并发连接数，挂死自动降级 |
| `--stall-secs` | 600 | 字节无增长判定挂死 |
| `--per-chunk-budget` | 0（不限） | 每 chunk 时间预算；GB 级 feature 保持 0，靠挂死看门狗 |
| `--min-free-gb` | 20 | 磁盘守卫阈值 |

示例：

```bash
# 只补 100-199 这段 egomotion
bash start_download.sh start --feature egomotion --chunks 100-199

# 下载标定（KB 级，几分钟完事）
bash start_download.sh start --feature camera_intrinsics

# 下载前置相机（GB 级/chunk，全量约十几 TB，先确认磁盘！）
bash start_download.sh start --feature camera_front_wide_120fov --min-free-gb 200
```

## 注意

- 环境要求：仓库 venv（`run_download.sh` 里 `PY` 指向 `.venv/bin/python`）、`~/.cache/huggingface/token`、镜像变量（脚本内已默认 `HF_ENDPOINT=https://hf-mirror.com`，无需额外设置）。
- 大文件 feature（相机/lidar）：单 chunk 数 GB、镜像单连接约 150-200KB/s，全量耗时以周计；并发调高会更快但更容易触发挂死（会自动降回来）。
- 下载缓存（`.cache/huggingface/download`）在 `<local_dir>` 下；本工具只清 `.incomplete/.lock`，不动已完成的 metadata。
