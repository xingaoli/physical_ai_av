# Meta-Action Auto-Labeling System

基于 CoC (Chain of Causality) 论文的 meta-action 自动标注和关键帧选择系统。

## 📋 功能概述

### 1. Meta-Action 检测
- **纵向**: Stop, Maintain speed, Gentle accelerate, Strong accelerate, Gentle decelerate, Strong decelerate, Reverse
- **横向**: Go straight, Steer left, Steer right, Sharp steer left, Sharp steer right, Reverse left, Reverse right

### 2. 关键帧选择
根据论文方法，将 meta-action 发生转变的时刻作为关键帧（决策时刻）。

### 3. 自动标注流程
- 输入: egomotion 数据 (速度、加速度、曲率)
- 处理: 信号平滑 → meta-action 分类 → 转变点检测
- 输出: JSON 标注文件 + 可视化图表

## 🚀 快速开始

### 前置条件
确保已经运行了 `tools/1_extract_egomotion.py` 生成 egomotion 数据。

### 基本使用

#### 1. 生成 Meta-Action 标注
```bash
# 处理单个 chunk
python3 tools/3_meta_action_annotation.py --chunks chunk_0000

# 处理多个 chunks
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 chunk_0001 chunk_0002

# 处理所有 chunks
python3 tools/3_meta_action_annotation.py

# 生成可视化（每个 chunk 的第一个视频）
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --viz
```

#### 2. 查看单个视频可视化
```bash
python3 tools/visualize_meta_actions.py chunk_0000 <video_uuid>
```

示例：
```bash
python3 tools/visualize_meta_actions.py chunk_0000 86de1c0c-e9cd-44ef-aad2-211c6b8a00da.egomotion
```

#### 3. 分析统计信息
```bash
# 分析指定 chunk
python3 tools/analyze_meta_actions.py --chunks chunk_0000 --plot

# 分析所有已处理的 chunks
python3 tools/analyze_meta_actions.py --plot
```

## 📊 输出结果

### 1. 标注 JSON 文件
位置: `data_dir/labels/meta_actions/meta_actions.{chunk_name}.json`

```json
[
  {
    "video_uuid": "86de1c0c-e9cd-44ef-aad2-211c6b8a00da",
    "num_frames": 200,
    "num_keyframes": 15,
    "keyframes": [
      {
        "frame_index": 0,
        "timestamp_us": 0,
        "timestamp_sec": 0.0,
        "long_action": "Maintain speed",
        "lat_action": "Go straight",
        "speed": 11.892,
        "acceleration": -0.072,
        "curvature": -0.000859
      }
    ],
    "action_statistics": {
      "longitudinal": {"Maintain speed": 120, "Stop": 30},
      "lateral": {"Go straight": 150, "Steer left": 30}
    },
    "keyframe_indices": {
      "longitudinal": [0, 45, 120],
      "lateral": [0, 30, 80],
      "combined": [0, 30, 45, 80, 120]
    }
  }
]
```

### 2. 可视化图表
- `{video_uuid}_meta_actions.png`: 单个视频的 meta-action 时间线
- `meta_action_statistics.png`: 整体统计图表（饼图、柱状图）

## ⚙️ 配置参数

### 默认阈值
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `strong_accel_threshold` | 2.0 m/s² | 强加速阈值 |
| `gentle_accel_threshold` | 0.5 m/s² | 轻微加速阈值 |
| `gentle_decel_threshold` | -0.5 m/s² | 轻微减速阈值 |
| `strong_decel_threshold` | -2.0 m/s² | 强减速阈值 |
| `stop_speed_threshold` | 0.5 m/s | 停车速度阈值 |
| `sharp_steer_threshold` | 0.005 1/m | 急转弯曲率阈值 |
| `gentle_steer_threshold` | 0.001 1/m | 轻微转向曲率阈值 |
| `acceleration_window` | 5 frames | 加速度平滑窗口 |
| `curvature_window` | 5 frames | 曲率平滑窗口 |

### 自定义配置

1. 复制配置模板：
```bash
cp tools/meta_action_config.example.json my_config.json
```

2. 修改阈值参数

3. 使用自定义配置：
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --config my_config.json
```

## 📈 实际运行示例

```bash
$ python3 tools/3_meta_action_annotation.py --chunks chunk_0000

================================================================================
Processing chunk_0000...
  chunk_0000: 100%|████████████| 100/100 [00:01<00:00, 99.90it/s]
  ✓ Saved 100 annotations to meta_actions.chunk_0000.json

================================================================================
SUMMARY STATISTICS
================================================================================
Total videos processed: 100
Total frames: 20,000
Total keyframes: 1,864
Average keyframes per video: 18.64
Compression ratio: 10.73x

Longitudinal action distribution:
  Maintain speed: 6771 (33.9%)
  Reverse: 4577 (22.9%)
  Gentle decelerate: 3059 (15.3%)
  Gentle accelerate: 2710 (13.6%)
  Stop: 1851 (9.3%)
  Strong decelerate: 644 (3.2%)
  Strong accelerate: 388 (1.9%)

Lateral action distribution:
  Go straight: 7629 (38.1%)
  Steer right: 2641 (13.2%)
  Steer left: 2519 (12.6%)
  Reverse left: 2263 (11.3%)
  Sharp steer right: 1859 (9.3%)
  Reverse: 1526 (7.6%)
  Reverse right: 788 (3.9%)
  Sharp steer left: 775 (3.9%)

================================================================================
✓ Meta-action annotation complete!
Output saved to: /path/to/data/labels/meta_actions
================================================================================
```

## 📁 文件结构

```
tools/
├── 1_extract_egomotion.py              # 提取 egomotion 数据
├── 2_extract_camera_frames.py          # 提取相机帧
├── 3_meta_action_annotation.py         # Meta-action 标注（主程序）
├── visualize_meta_actions.py           # 单视频可视化
├── analyze_meta_actions.py             # 统计分析
└── meta_action_config.example.json     # 配置模板

docs/
├── meta_action.md                      # 论文原文描述
└── meta_action_usage.md                # 详细使用说明

data_dir/labels/
├── egomotion_corrected/                # 输入: egomotion 数据
│   └── egomotion.chunk_*.zip
└── meta_actions/                       # 输出: meta-action 标注
    ├── meta_actions.chunk_*.json       # 标注数据
    ├── *_meta_actions.png              # 单视频可视化
    └── meta_action_statistics.png      # 统计图表
```

## 🔬 技术细节

### Meta-Action 检测规则

#### 纵向检测
1. **Reverse**: `vy < -0.5 m/s`
2. **Stop**: `speed < 0.5 m/s`
3. **Strong accelerate**: `acceleration > 2.0 m/s²`
4. **Gentle accelerate**: `0.5 < acceleration ≤ 2.0 m/s²`
5. **Maintain speed**: `-0.5 ≤ acceleration ≤ 0.5 m/s²`
6. **Gentle decelerate**: `-2.0 ≤ acceleration < -0.5 m/s²`
7. **Strong decelerate**: `acceleration < -2.0 m/s²`

#### 横向检测
1. **Reverse left**: `vy < -0.5 AND curvature < -0.001`
2. **Reverse right**: `vy < -0.5 AND curvature > 0.001`
3. **Sharp steer left**: `curvature < -0.005`
4. **Sharp steer right**: `curvature > 0.005`
5. **Steer left**: `-0.005 ≤ curvature < -0.001`
6. **Steer right**: `0.001 < curvature ≤ 0.005`
7. **Go straight**: `-0.001 ≤ curvature ≤ 0.001`

### 信号处理
- 使用中值滤波器 (median filter) 平滑信号
- 窗口大小: 5 帧 (0.5 秒)
- 保留边缘的同时减少噪声

### 关键帧选择
- 检测 meta-action 标签转变点
- 默认包含首帧和末帧
- 合并纵向和横向转变点（去重）

## 🎯 与论文的对应

| 论文描述 | 实现状态 |
|---------|---------|
| 原子 meta-actions (Table 5) | ✅ 全部实现 (7×7) |
| 10Hz 帧级别标注 | ✅ 10Hz采样 |
| 基于转变的关键帧选择 | ✅ transition detection |
| 纵向+横向独立处理 | ✅ 分别检测再合并 |

## 📝 使用场景

1. **训练数据准备**: 为 CoC 模型自动生成标注
2. **数据分析**: 分析驾驶行为模式
3. **关键帧提取**: 压缩视频数据（10.73x 压缩比）
4. **行为理解**: 理解车辆决策时刻

## 🔧 故障排除

### 问题 1: 找不到 egomotion 数据
```
Error: No egomotion chunks found
```
解决: 先运行 `tools/1_extract_egomotion.py`

### 问题 2: 可视化失败
```
Warning: Visualization failed
```
解决: 安装 matplotlib `pip install matplotlib`

### 问题 3: 检测结果不准确
解决: 调整配置文件中的阈值参数，重新运行

## 📚 参考资料

- 论文: Chain of Causality (CoC)
- Table 5: Meta-Actions 定义
- Section 4.3.2: Auto-Labeling 方法

## 🤝 贡献

欢迎提交 PR 和 Issue！

## 📄 许可证

与主项目保持一致

---

**作者**: Physical AI AV Team
**日期**: 2026-02-06
**版本**: 1.0.0
