# Meta-Action Auto-Labeling 使用说明

## 概述

基于 CoC 论文的方法，实现了通过车辆运动数据自动标注 meta-action 并选择关键帧的系统。

### Meta-Actions 定义（论文 Table 5）

#### 纵向
- **Stop**: 速度接近 0 (< 0.5 m/s)
- **Maintain speed**: 加速度接近 0，速度不为 0
- **Gentle accelerate**: 小正加速度 (0.5 - 2 m/s²)
- **Strong accelerate**: 大正加速度 (> 2 m/s²)
- **Gentle decelerate**: 小负加速度 (-0.5 到 -2 m/s²)
- **Strong decelerate**: 大负加速度 (< -2 m/s²)
- **Reverse**: 倒车 (vy < -0.5 m/s)

#### 横向
- **Go straight**: 曲率接近 0
- **Steer left**: 小左转 (曲率 < -0.001)
- **Steer right**: 小右转 (曲率 > 0.001)
- **Sharp steer left**: 大左转 (曲率 < -0.005)
- **Sharp steer right**: 大右转 (曲率 > 0.005)
- **Reverse left/reverse right**: 倒车时转向

## 使用方法

### 1. 基本使用

处理单个 chunk：
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000
```

处理多个 chunks：
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 chunk_0001 chunk_0002
```

处理所有 chunks：
```bash
python3 tools/3_meta_action_annotation.py
```

### 2. 生成可视化

为每个 chunk 的第一个视频生成可视化图：
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --viz
```

可视化图包含：
- 速度曲线
- 加速度曲线 + 纵向 meta-action 背景
- 曲率曲线 + 横向 meta-action 背景
- 关键帧标记

### 3. 自定义阈值

创建配置文件（参考 `meta_action_config.example.json`）：
```json
{
  "strong_accel_threshold": 2.0,
  "gentle_accel_threshold": 0.5,
  "gentle_decel_threshold": -0.5,
  "strong_decel_threshold": -2.0,
  "stop_speed_threshold": 0.5,
  "sharp_steer_threshold": 0.005,
  "gentle_steer_threshold": 0.001,
  "acceleration_window": 5,
  "curvature_window": 5
}
```

使用自定义配置：
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --config my_config.json
```

### 4. 指定输出目录

```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --output-dir /path/to/output
```

## 输出格式

### JSON 文件结构

输出文件: `meta_actions.{chunk_name}.json`

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
        "long_action": "Stop",
        "lat_action": "Go straight",
        "speed": 0.0,
        "acceleration": 0.1,
        "curvature": 0.0
      },
      ...
    ],
    "action_statistics": {
      "longitudinal": {
        "Maintain speed": 120,
        "Gentle accelerate": 50,
        "Stop": 30
      },
      "lateral": {
        "Go straight": 150,
        "Steer left": 30,
        "Steer right": 20
      }
    },
    "keyframe_indices": {
      "longitudinal": [0, 45, 120, 180, 199],
      "lateral": [0, 30, 80, 150, 199],
      "combined": [0, 30, 45, 80, 120, 150, 180, 199]
    }
  }
]
```

## 关键帧选择策略

根据论文描述：
> "we treat the frame at which a meta action transition occurs as a decision-making moment"

关键帧 = meta-action 发生转变的时刻

- **纵向关键帧**: 纵向 meta-action 改变的帧
- **横向关键帧**: 横向 meta-action 改变的帧
- **组合关键帧**: 纵向或横向任一改变的时刻（并集）

默认策略：
- 包含第一帧 (frame 0)
- 包含最后一帧
- 去除重复（同一帧既是纵向又是横向转变）

## 统计信息示例

运行完成后会输出：
```
================================================================================
SUMMARY STATISTICS
================================================================================
Total videos processed: 17
Total frames: 3400
Total keyframes: 255
Average keyframes per video: 15.00
Compression ratio: 13.33x

Longitudinal action distribution:
  Maintain speed: 1800 (52.9%)
  Gentle accelerate: 800 (23.5%)
  Stop: 400 (11.8%)
  Gentle decelerate: 300 (8.8%)
  Strong accelerate: 100 (2.9%)

Lateral action distribution:
  Go straight: 2500 (73.5%)
  Steer left: 500 (14.7%)
  Steer right: 400 (11.8%)
================================================================================
```

## 数据要求

输入数据格式（egomotion parquet 文件）：
- `timestamp`: 时间戳 (微秒)
- `vx, vy, vz`: 速度 (m/s)
- `ax, ay, az`: 加速度 (m/s²)
- `curvature`: 曲率 (1/m)

采样频率: 10 Hz (每 100ms 一帧)
场景时长: 20 秒 (200 帧)

## 实现细节

### 信号处理
- 使用中值滤波器 (median filter) 平滑加速度和曲率信号
- 窗口大小默认为 5 帧 (0.5 秒)
- 保留边缘的同时减少噪声

### 检测逻辑
1. **预处理**: 计算速度、加速度，应用平滑滤波
2. **纵向检测**: 基于速度和加速度阈值分类
3. **横向检测**: 基于曲率阈值分类，考虑倒车情况
4. **关键帧选择**: 检测 label 转变点

### 优化参数
- `min_transition_gap`: 最小关键帧间隔 (默认 5 帧)
- 避免在短时间内重复检测同一转变

## 与论文的对应

✓ **原子 meta-actions**: 实现了 Table 5 的所有 7×7 个类别
✓ **10Hz 标注**: 在帧级别自动标注
✓ **关键帧选择**: meta-action 转变时刻
✓ **纵向+横向**: 分别处理并合并

## 下一步

1. 将标注结果用于 CoC (Chain of Causality) 模型训练
2. 分析不同场景的 meta-action 模式
3. 根据实际数据调整阈值参数
4. 扩展到更高级的驾驶决策标注
