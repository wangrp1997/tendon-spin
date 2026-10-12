# TendonSpin

伯牙腱驱灵巧手的手内物体旋转项目。使用 [Hora](https://github.com/HaozhiQi/hora) PPO 训练教师策略，
研究触觉与动作历史驱动的学生策略；参考 [Hora](https://haozhi.io/hora)、
[AnyRotate](https://maxyang27896.github.io/anyrotate/) 和
[Sharpa RL Lab](https://github.com/sharpa-robotics/sharpa-rl-lab)。

代码和模型资源独立保存在本仓库。仿真使用现有 Isaac Sim/Lab 环境，
同时保留原始 MuJoCo 模型与工具。

## 目录

| 目录 | 内容 |
| --- | --- |
| `tendonspin/` | 手模型接口、仿真环境、策略和评价指标 |
| `scripts/` | 训练、冻结策略评估和视频渲染入口 |
| `assets/` | CAD、原始抓取状态和可选物体配置 |
| `third_party/` | 参考算法源码及许可证 |
| `docs/` | 适配说明、实验协议和结果 |
| `outputs/` | 本地权重、TensorBoard、轨迹和视频，不提交到 Git |

## 使用

在已有 Isaac Lab Python 环境中运行，项目根目录加入 `PYTHONPATH`：

```bash
PYTHONPATH=. python scripts/train_boya_hora.py --help
PYTHONPATH=. python scripts/evaluate_boya_hora_object.py --help
```

物体默认使用原始 40×32 毫米圆柱，也可选择命名配置或指定圆柱、球、长方体的尺寸。
参数见[物体配置说明](docs/object_geometry.md)。

## 文档

- [Hora 到 Boya 的适配](docs/hora_boya_port.md)
- [参考代码与触觉接口](docs/reference_reuse.md)
- [实验记录与索引](docs/experiment_state.md)
- [部署接口](docs/deployment.md)
