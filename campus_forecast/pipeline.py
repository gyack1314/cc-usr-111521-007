"""训练流水线：历史回放校准 + 全量可见资料训练，产出生产模型工件。"""
from .backtest import BacktestConfig, fill_actuals, run_backtest
from .dataset import OriginSnapshot
from .historical import historical_calibration
from .modeling import train_model


def train_production_model(book, origin, model_id, backtest_origins, horizons,
                           status=None, use_historical_calibration=True):
    """走查回测评估 + 历史回放校准 + 在 origin 全量可见资料上训练。

    - 回测记录用于向运营方披露滚动 origin 的分层误差与覆盖，不参与模型拟合；
    - 区间校准来自历史回放的样本外残差（每个折叠只看当时可见资料）；
    - 最终模型在 origin 的单一全量快照上训练，校准表随工件一起保存。
    """
    def log(message):
        if status is not None:
            status(message)

    config = BacktestConfig(origins=list(backtest_origins),
                            horizons=tuple(horizons))
    records, model_info = run_backtest(book, config, status=log)
    fill_actuals(book, records)

    if use_historical_calibration:
        calibration, residual_count = historical_calibration(
            book, origin, status=log)
        calibration_source = 'historical-replay'
    else:
        realized = [r for r in records if r.residual_log is not None]
        if not realized:
            raise ValueError('回测没有任何可核对的预测，无法校准区间')
        from .calibration import calibrate
        calibration = calibrate(
            [(r.cohort, r.bucket, r.residual_log) for r in realized])
        residual_count = len(realized)
        calibration_source = 'walkforward'

    snapshot = OriginSnapshot(book, origin)
    model = train_model([snapshot], model_id, origin, calibration=calibration)
    model.training['backtest_origins'] = sorted(set(config.origins))
    model.training['backtest_models'] = model_info
    model.training['calibration_residuals'] = residual_count
    model.training['calibration_source'] = calibration_source
    return model, records
