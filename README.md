# 零碳校园负荷的冷启动预测与区间交付资料工具

这个 Python 项目保存业务记录及其披露修订，在按发布时点取数的基础上，提供
可保存、可复用的多时段负荷预测服务：点预测、边际预测区间、适用范围与降级/
拒答说明。样本为虚构资料，仅用于核对字段与流程，不代表真实机构情况。

## 文件与运行

- `campus_forecast/records.py`：不可变资料副本、修订选择和指纹。
- `campus_forecast/dataset.py`：发布时点快照 `OriginSnapshot`，所有可见性判断的唯一入口。
- `campus_forecast/features.py`：固定特征列、标准化统计、距离桶与人群划分、缺失中性填补。
- `campus_forecast/modeling.py`：纯标准库岭回归（Cholesky 求解）与层次化冷启动模型、模型工件。
- `campus_forecast/calibration.py`：残差分位数校准、小样本稳健构造与分层回退/拒答。
- `campus_forecast/historical.py`：无泄漏的历史回放校准（稀疏折叠、样本外残差）。
- `campus_forecast/backtest.py`：走查式回测与「人群×距离桶×逐校」分层评估。
- `campus_forecast/service.py`：预测服务与 `ForecastStore` 工件存储。
- `campus_forecast/pipeline.py`：回测评估 + 历史校准 + 全量训练的生产流水线。
- `campus_forecast/cli.py` / `__main__.py`：命令行入口。
- `data/sample.json`：可直接读取的业务样本。

Python 3.11 或更新版本即可，不依赖第三方包。

```sh
# 测试与编译检查
python -m unittest discover -s tests -v
python -m compileall -q campus_forecast

# 历史诊断（保留原有行为）
python -m campus_forecast data/sample.json --as-of 2026-11-15T00:00:00Z

# 训练：走查回测评估 + 历史回放校准 + 全量可见资料训练，模型写入 store
python -m campus_forecast train data/sample.json \
  --origin 2026-11-04T00:00:00+00:00 --model-id prod-001 --store ./store

# 多时段预测（点预测 + 区间 + 适用范围 + 降级说明），预测工件写入 store
python -m campus_forecast forecast data/sample.json --store ./store \
  --model-id prod-001 --origin 2026-11-04T00:00:00+00:00 --school new \
  --horizons 1 2 6 12 24

# 走查回测并输出分层误差与区间覆盖（可加 --out 保存全部预测明细）
python -m campus_forecast backtest data/sample.json --days 14

# 工件清单与复算一致性
python -m campus_forecast list --store ./store
python -m campus_forecast replay data/sample.json --store ./store \
  --forecast-id <id>
```

## 发布时点纪律（无泄漏）

训练、回测、实际预测只能使用各自发布时刻已经可见的信息，由 `OriginSnapshot`
统一强制：

- 负荷/校历/学校属性按 `known_at` 截断；天气额外要求 `issued_at <= origin`，
  气象预报修订在发布前不可见。
- 电表迟到（样本中 3–62 小时）与修订（revision 2）只在其 `known_at` 之后进入视图；
  标签未知与负荷为零严格区分，缺失特征用中性值 + 缺失指示位表达，绝不填零。
- 走查回测的每个 origin 独立训练，只看当时快照；标签在后续时点才核对，
  未到期的预测保留为 pending，不回填、不参与评估。
- 历史回放校准的每个折叠模型只用折叠 origin 之前的标签拟合，对折叠之后的
  目标计算样本外残差；新学校在任何折叠中都读不到自己的未来结果。

## 冷启动建模与区间

模型在 `log1p(kWh)` 空间分三层，全部确定性、可逐位复现：

1. **校级水平**：每校已可见标签均值，按经验贝叶斯向全局水平部分池化；
   零历史学校回退全局水平，有 1–2 天读数即可获得自己的规模估计。
2. **全局形态模型**：在所有学校的去水平残差上做岭回归，学习小时曲线、
   课型、气温、气候带与预测距离效应，是新学校唯一借用的共享先验。
3. **校级残差模型**：仅当本校样本达到阈值才拟合，预测时按可见历史量与
   全局形态收缩混合；训练按 horizon 屏蔽结构上不可见的短期滞后（滞后 L
   仅当 L≥horizon 时不晚于 origin），训练、回测与实际预测使用完全相同的
   缺失条件。所有缺失滞后的中性填补只取严格早于目标小时的可见负荷，
   绝不包含该行自身标签或相对目标的未来小时。

区间为**逐点边际覆盖（非联合覆盖）**，在 log1p 空间按「人群 × 距离桶」
用样本外残差的经验分位数校准；样本不足时用中位数/MAD 稳健构造并按小样本
t 倍数加宽，仍不足则按「同人群汇总 → 全局」回退并标注来源，
连全局池都不足或气象缺失时**拒答区间**（点预测仍可给出）。

## 分层评估，不允许只看平均数

回测报告分别给出总体、`new/seen × h1/h2_6/h7_24/h25_plus` 人群距离桶、
逐学校距离桶、以及模型层级的 MAE/RMSE/MAPE/偏差与 80%、95% 区间覆盖率、
平均宽度和拒答数。样本资料上的代表性结果（14 个每日 origin，无泄漏回放）：

| 人群 | 距离桶 | MAPE | 95% 区间覆盖（留一折叠） | 说明 |
| --- | --- | --- | --- | --- |
| 已见学校 | h7_24（次日全天） | 0.057 | 0.935 | 主交付桶，覆盖接近名义水平 |
| 新学校 | h7_24 | 0.125 | 0.804 | 仅 5 天历史，离散度大、偏保守 |
| 任意人群 | h25_plus（超气象支持） | — | 区间拒答 | 停止输出不可依赖的区间 |
| 新学校 | 零历史首日 | 偏差最大 | 宽区间/稳健标记 | 冷启动固有不确定性，不借用未来数据掩盖 |

## 模型工件与可复现性

模型工件是单个 JSON，包含特征版本与完整列顺序、标准化统计、全局/校级
系数、部分池化参数与完整校准表。加载时校验 `feature_version` 与特征列，
不匹配直接拒绝。重新训练在相同资料下产生逐位相同的系数；已保存预测工件
可用 `replay` 复算，点预测与区间必须完全一致。工件只增不改，重复保存会被拒绝。

## 降级与适用范围

每个预测行带 `status`（`ok`/`degraded`/`refused`）与原因，整份工件带
`applicability`：人群、可见历史小时数、各人群已校准距离桶、气象实际支持
的最大预测距离（按快照动态计算，而非写死）与边际覆盖声明。气象缺失、
校准池不足、零历史等情形都会显式说明依据，而不是输出无法负责的精确数字。

领域单位与兼容语义见 [CONTRACT.md](CONTRACT.md)。
