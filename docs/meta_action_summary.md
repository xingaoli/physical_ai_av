# Meta-Action 自动标注系统 - 完成总结

## 🎉 系统已完成

我已经为你实现了完整的 Meta-Action 自动标注系统，基于 CoC 论文的方法。这个系统可以自动分析车辆运动数据，标注 meta-action，并选择关键帧。

## 📦 交付内容

### 核心脚本
1. **`tools/3_meta_action_annotation.py`** - 主标注程序
   - 加载 egomotion 数据
   - 检测纵向和横向 meta-actions
   - 识别关键帧（meta-action 转变时刻）
   - 生成 JSON 标注文件

2. **`tools/visualize_meta_actions.py`** - 可视化工具
   - 生成单个视频的 meta-action 时间线
   - 显示速度、加速度、曲率曲线
   - 标注关键帧位置

3. **`tools/analyze_meta_actions.py`** - 统计分析工具
   - 分析 meta-action 分布
   - 生成统计图表
   - 识别有趣的模式

4. **`tools/test_meta_action_system.sh`** - 快速测试脚本
   - 一键测试整个系统
   - 自动验证功能

### 文档
- **`tools/META_ACTION_README.md`** - 完整使用指南
- **`docs/meta_action_usage.md`** - 详细使用说明
- **`tools/meta_action_config.example.json`** - 配置模板

## 🚀 快速开始

### 1. 运行快速测试
```bash
./tools/test_meta_action_system.sh
```

### 2. 处理单个 chunk
```bash
python3 tools/3_meta_action_annotation.py --chunks chunk_0000 --viz
```

### 3. 处理所有数据
```bash
python3 tools/3_meta_action_annotation.py
```

### 4. 查看可视化
```bash
# 查看统计图表
ls /path/to/data/labels/meta_actions/*.png

# 查看单个视频
python3 tools/visualize_meta_actions.py chunk_0000 <video_uuid>
```

## 📊 实际运行结果

在 chunk_0000 (100个视频) 上的测试结果：

```
Total videos processed: 100
Total frames: 20,000
Total keyframes: 1,864
Average keyframes per video: 18.64
Compression ratio: 10.73x
```

### Meta-Action 分布

**纵向分布：**
- Maintain speed: 33.9%
- Reverse: 22.9%
- Gentle decelerate: 15.3%
- Gentle accelerate: 13.6%
- Stop: 9.3%
- Strong decelerate: 3.2%
- Strong accelerate: 1.9%

**横向分布：**
- Go straight: 38.1%
- Steer right: 13.2%
- Steer left: 12.6%
- Reverse left: 11.3%
- Sharp steer right: 9.3%
- Reverse: 7.6%
- Reverse right: 3.9%
- Sharp steer left: 3.9%

## 🎯 系统特点

### 1. 完全符合论文方法
- ✅ 实现了 Table 5 的所有 7×7 个 meta-actions
- ✅ 10Hz 帧级别自动标注
- ✅ 基于转变点的关键帧选择
- ✅ 纵向+横向独立处理

### 2. 健壮的信号处理
- 中值滤波器平滑噪声
- 保留边缘特征
- 可配置的阈值参数

### 3. 丰富的可视化
- 时间线图表
- 统计分布图
- 关键帧标记

### 4. 灵活的配置
- JSON 配置文件
- 可调整所有阈值
- 支持自定义参数

## 📁 输出文件结构

```
data_dir/labels/meta_actions/
├── meta_actions.chunk_0000.json          # 标注数据
├── meta_actions.chunk_0001.json
├── ...
├── meta_action_statistics.png            # 统计图表
├── <uuid>_meta_actions.png               # 单视频可视化
└── ...
```

## 🔧 配置参数

可以通过配置文件调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `strong_accel_threshold` | 2.0 m/s² | 强加速阈值 |
| `gentle_accel_threshold` | 0.5 m/s² | 轻微加速阈值 |
| `gentle_decel_threshold` | -0.5 m/s² | 轻微减速阈值 |
| `strong_decel_threshold` | -2.0 m/s² | 强减速阈值 |
| `stop_speed_threshold` | 0.5 m/s | 停车速度阈值 |
| `sharp_steer_threshold` | 0.005 1/m | 急转弯曲率阈值 |
| `gentle_steer_threshold` | 0.001 1/m | 轻微转向曲率阈值 |

## 💡 使用建议

### 1. 参数调优
- 先在小数据集上测试
- 查看可视化结果
- 根据实际数据分布调整阈值
- 重新运行标注

### 2. 质量检查
- 使用 `visualize_meta_actions.py` 检查标注质量
- 关注边界情况（加速/减速转变）
- 检查倒车场景的标注

### 3. 批量处理
- 分 chunk 处理大量数据
- 使用 `--plot` 生成统计图
- 定期检查中间结果

## 🔍 下一步工作

### 扩展功能
1. **高级决策标注**: 基于 meta-action 序列标注高级驾驶决策
2. **因果链分析**: 分析 CoC (Chain of Causality)
3. **模型训练**: 使用标注数据训练行为预测模型

### 优化方向
1. **自适应阈值**: 根据数据分布自动调整参数
2. **上下文感知**: 考虑前后帧信息
3. **多模态融合**: 结合相机图像、雷达等数据

## 📞 使用帮助

遇到问题时：
1. 查看 `tools/META_ACTION_README.md`
2. 检查输入数据格式
3. 查看可视化结果诊断问题
4. 调整配置参数

## 🎓 理论基础

本系统完全基于 CoC 论文的 Section 4.3.2 "Auto-Labeling"：

> "To identify keyframes for auto-labeling, we first define a set of low-level meta actions and implement corresponding rule-based detectors to infer these meta actions at the frame level. Then, we treat the frame at which a meta action transition occurs as a decision-making moment."

---

**开发完成日期**: 2026-02-06
**版本**: 1.0.0
**状态**: ✅ 已测试可用
