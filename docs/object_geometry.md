# 物体尺寸与形状参数

冻结策略评估入口 `scripts/evaluate_boya_hora_object.py` 支持以下参数。
原训练入口与已完成实验不变；这些参数目前用于评估和对应轨迹的视频渲染。

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--object-profile boya_original` | 原始圆柱：直径40、轴向高度32毫米，质量50克 |
| `--object-profile sharpa_recommended` | Sharpa README推荐圆柱：直径48、高度60毫米；质量仍为Boya的50克 |
| `--object-shape cylinder\|sphere\|box` | 覆盖形状 |
| `--object-diameter-mm` | 圆柱或球的直径，毫米 |
| `--object-height-mm` | 圆柱的轴向高度，毫米 |
| `--object-size-mm X Y Z` | 长方体的三个边长，毫米；选择box时必须提供 |
| `--object-mass-g` | 质量，克 |

向完整评估命令添加以下任选一组参数即可；调用帮助可查看检查点、缓存及输出等必需参数：

```bash
PYTHONPATH=. python scripts/evaluate_boya_hora_object.py --help

# 保留原始尺寸（不传物体参数时也是此配置）
--object-profile boya_original

# Sharpa推荐的实物圆柱尺寸
--object-profile sharpa_recommended

# 自定义圆柱 / 球 / 长方体
--object-shape cylinder --object-diameter-mm 40 --object-height-mm 32
--object-shape sphere --object-diameter-mm 48
--object-shape box --object-size-mm 40 40 60
```

配置在[profiles.json](../assets/objects/profiles.json)。尺寸和质量必须为有限正数。
渲染器读取评估记录中的同一配置，旧记录缺省为原始40×32毫米圆柱。

改变几何不会自动调整原始手姿、物体中心、朝向、操作区边界或教师观测中的形状常量，
也不会重新训练策略。较大或不同形状的物体可能与原始抓取不兼容；此时直接评估失败
就是迁移结果，不能据此声称已适配该物体。后续训练需要另行声明相应抓取、缓存与配置。

Sharpa尺寸出处及仿真/实物差异见[尺寸核对](research/2026-10-12-sharpa-object-size.md)。
本次B40M单次尺寸迁移的边界见[评估协议](experiments/2026-10-12-b40m-sharpa-size/PROTOCOL.md)。
