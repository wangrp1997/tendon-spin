# 同条件教师对照

相同原始grasp44、boya_workspace终止、120s预算、seed43；各自单独初始化，无重置或控制器切换。

| 权重 | 窗口s | 净角° | 有效s | 峰角° | 倒转° | 停止原因 |
|---|---:|---:|---:|---:|---:|---|
| old_weights_new_criteria | 30 | 142.646 | 3.3495 | 184.816 | 43.464 | object below Boya manipulation region |
| old_weights_new_criteria | 120 | 142.646 | 3.3495 | 184.816 | 43.464 | object below Boya manipulation region |
| new_weights_new_criteria | 30 | 182.936 | 5.5995 | 182.936 | 1.266 | object below Boya manipulation region |
| new_weights_new_criteria | 120 | 182.936 | 5.5995 | 182.936 | 1.266 | object below Boya manipulation region |

新权重行复用其原定独立评估，旧权重行是新的同条件执行；不是旧轨迹重计分。
位移、倾斜、力极值与末30s推进见JSON/CSV。新旧窗口均包含所有早停，不拼接角度。
原旧规则63.626313°仅保留在历史记录，不混入本表。单初态对照不证明SOTA或统计显著性。
