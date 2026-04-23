#!/usr/bin/env python3
# ============================================================================
# 数据下载脚本（实际下载逻辑）
# ============================================================================
# 调用关系：
#   start_download_scheduled.sh（包装脚本，PID管理和nohup启动）
#     └─> download_scheduled.sh（定时控制和重试逻辑）
#           └─> download_data.py（本脚本，实际下载）
#
# 功能：
#   - 下载指定范围的 chunk 数据（默认 chunk 50-199）
#   - 下载 calibration、labels、camera 等特征数据
#   - 使用 HuggingFace 镜像站下载（hf-mirror.com）
#   - 禁用 Xet 协议避免超时问题
#
# 使用方式：
#   - 通过 download_data_retry.sh 调用（推荐）
#   - 或直接运行：python download_data.py --chunk_id -1 --local_dir /path/to/data
#   - 下载指定chunk：python download_data.py --chunk_id 50,51,52 --local_dir /path/to/data
# ============================================================================

# 必须在导入 huggingface_hub 之前设置环境变量！
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'  # 禁用 Xet 协议（避免 cas-bridge.xethub.hf.co 超时）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像站

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
# 在导入后再次确保 Xet 被禁用（防止模块加载时环境变量未生效）
import huggingface_hub.constants
huggingface_hub.constants.HF_HUB_DISABLE_XET = True

import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description='download data')
parser.add_argument('--chunk_id', type=str, help='download chunk_id')
parser.add_argument('--local_dir', type=str, help='download local_dir')

args = parser.parse_args()

if args.chunk_id == "-1":
    chunk_id_list = list(np.arange(0, 50))
else:
    chunk_id_list = [int(x) for x in args.chunk_id.split(',')]

print(f"准备下载 chunks: {list(chunk_id_list) if len(chunk_id_list) <= 10 else f'{list(chunk_id_list[:5])}...{list(chunk_id_list[-5:])}'}")

# chunk_id_list = args.chunk_id.split(',')
local_dir = args.local_dir
# 禁用下载大小确认提示（后台运行无法交互式输入）
ds = PhysicalAIAVDatasetInterface(
    token=True,
    local_dir=local_dir,
    confirm_download_threshold_gb=float("inf"),
)

failed_chunks = []

for chunk_id in chunk_id_list:
    chunk_id = int(chunk_id)  # 确保是原生 int，不是 np.int64
    try:
        print(f"\n{'='*60}")
        print(f"开始下载 chunk {chunk_id}...")
        print(f"{'='*60}")

        # calibration data
        print(f"  [1/5] 下载 CALIBRATION...")
        ds.download_chunk_features(
            chunk_id,
            features=ds.features.CALIBRATION.ALL
        )

        # egomotion data
        print(f"  [2/5] 下载 LABELS (egomotion)...")
        ds.download_chunk_features(
            chunk_id,
            features=ds.features.LABELS.ALL
        )

        # ds.download_chunk_features(
        #     chunk_id,
        #     features=ds.features.RADAR.ALL
        # )

        # camera data
        print(f"  [3/5] 下载 CAMERA_CROSS_LEFT_120FOV...")
        ds.download_chunk_features(
            chunk_id=chunk_id,
            features=ds.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        )
        print(f"  [4/5] 下载 CAMERA_FRONT_WIDE_120FOV...")
        ds.download_chunk_features(
            chunk_id=chunk_id,
            features=ds.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        )
        print(f"  [5/5] 下载 CAMERA_CROSS_RIGHT_120FOV 和 CAMERA_FRONT_TELE_30FOV...")
        ds.download_chunk_features(
            chunk_id=chunk_id,
            features=ds.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        )
        ds.download_chunk_features(
            chunk_id=chunk_id,
            features=ds.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        )

        print(f"✅ chunk {chunk_id} 下载完成")

        # lidar data
        # ds.download_chunk_features(
        #     chunk_id,
        #     features=ds.features.LIDAR.LIDAR_TOP_360FOV
        # )

    except Exception as e:
        print(f"❌ chunk {chunk_id} 下载失败: {e}")
        import traceback
        traceback.print_exc()
        failed_chunks.append(chunk_id)
        # 继续下载下一个chunk，而不是立即退出
        # 这样可以让成功的chunk都下载完成
        continue

# 所有chunk处理完毕后，检查是否有失败的
if failed_chunks:
    print(f"\n{'='*60}")
    print(f"❌ 以下 chunk 下载失败: {failed_chunks}")
    print(f"{'='*60}")
    # 退出码1，触发重试逻辑
    sys.exit(1)
else:
    print(f"\n{'='*60}")
    print(f"✅ 所有 chunk 下载成功！")
    print(f"{'='*60}")
    sys.exit(0)
