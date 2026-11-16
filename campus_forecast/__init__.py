"""零碳校园负荷的冷启动预测与区间交付服务。

公开入口：
- RecordBook：不可变资料副本与发布时点筛选
- OriginSnapshot：某个 origin 的可见资料索引（无泄漏的唯一取数入口）
- ForecastModel / ForecastStore：模型工件与预测工件的保存、加载、复用
- forecast_school：多时段点预测 + 边际区间 + 适用范围 + 降级说明
- train_production_model：走查回测 + 历史回放校准 + 全量训练流水线
- run_backtest / evaluate：走查式回测与分层误差、区间覆盖评估
"""
from .records import RecordBook
from .dataset import OriginSnapshot
from .modeling import ForecastModel
from .service import ForecastStore, forecast_school
from .pipeline import train_production_model
from .backtest import run_backtest, evaluate

__all__ = [
    'RecordBook', 'OriginSnapshot', 'ForecastModel', 'ForecastStore',
    'forecast_school', 'train_production_model', 'run_backtest', 'evaluate',
]
