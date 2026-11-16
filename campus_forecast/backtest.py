"""走查式（walk-forward）回测与分层评估。

每个回测 origin 独立训练模型：训练快照只包含该 origin 已可见的资料，
预测目标是 origin 之后的小时，等这些小时的标签在后续 origin 可见后再核对。
评估按「人群（新/已见学校）× 预测距离桶」与逐学校分别报告误差和区间覆盖，
不允许只用总体平均数掩盖失效人群。
"""
import math
from dataclasses import dataclass

from .calibration import LEVELS, apply_interval, calibrate
from .dataset import OriginSnapshot, hour_add, instant
from .features import cohort_of, horizon_bucket
from .modeling import expm1, log1p, train_model

DEFAULT_HORIZONS = tuple(range(1, 25))


@dataclass
class BacktestConfig:
    origins: list
    horizons: tuple = DEFAULT_HORIZONS
    model_prefix: str = 'backtest'


@dataclass
class PredictionRecord:
    origin: str
    school: str
    target_time: str
    horizon: int
    cohort: str
    bucket: str
    point: float
    actual: float
    weather_available: bool
    known_hours: int
    tier: str
    blend_weight: float
    intervals: dict
    residual_log: float

    def to_dict(self):
        return {
            'origin': self.origin, 'school': self.school,
            'target_time': self.target_time, 'horizon': self.horizon,
            'cohort': self.cohort, 'bucket': self.bucket,
            'point': self.point, 'actual': self.actual,
            'weather_available': self.weather_available,
            'known_hours': self.known_hours, 'tier': self.tier,
            'blend_weight': self.blend_weight, 'intervals': self.intervals,
            'residual_log': self.residual_log}


def run_backtest(book, config, status=None):
    """执行走查回测，返回预测明细、校准表与每个 origin 的模型信息。

    每个 origin 的校准只使用此前 origin 已做出、且其标签在当前 origin
    已经可见的预测残差——标签可见性由当前快照直接判定，杜绝未来信息。
    """
    records = []
    model_info = []
    origins = sorted(instant(o) for o in config.origins)
    if not origins:
        raise ValueError('回测至少需要一个 origin')
    horizons = tuple(config.horizons)
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons) \
            or len(set(horizons)) != len(horizons):
        raise ValueError('预测距离须为互不相同的正整数小时')
    for index, origin in enumerate(origins):
        origin_text = origin.isoformat()
        snapshot = OriginSnapshot(book, origin_text)
        # 先给此前预测中"标签现已可见"的记录补残差，再据此校准当前模型
        residual_feed = []
        for prior in records:
            if prior.residual_log is not None:
                residual_feed.append((prior.cohort, prior.bucket, prior.residual_log))
                continue
            if instant(prior.target_time) >= origin:
                continue
            actual = snapshot.load_value(prior.school, prior.target_time)
            if actual is None:
                continue
            prior.actual = actual
            prior.residual_log = log1p(actual) - log1p(prior.point)
            residual_feed.append((prior.cohort, prior.bucket, prior.residual_log))
        model_id = f'{config.model_prefix}-{index:03d}-{origin.strftime("%Y%m%dT%H")}'
        model = train_model([snapshot], model_id, origin_text)
        model.calibration = calibrate(residual_feed) if residual_feed else {}
        model_info.append({
            'model_id': model_id, 'origin': origin_text,
            'sample_count': model.training['sample_count'],
            'local_schools': sorted(model.training['local_sample_counts']),
            'calibration_residuals': len(residual_feed),
            'fingerprint': snapshot.fingerprint})
        builder = model.builder()
        for school in sorted(snapshot.schools):
            known_hours = snapshot.known_hours(school)
            cohort = cohort_of(known_hours)
            for horizon in config.horizons:
                target_time = hour_add(origin_text, horizon)
                vector = builder.vector(snapshot, school, target_time, horizon)
                point_log, tier, weight = model.predict_log(vector, school, known_hours)
                point = expm1(point_log)
                weather = snapshot.weather_for(school, target_time)
                intervals = apply_interval(
                    model.calibration, cohort, horizon, point_log,
                    weather_available=weather is not None)
                records.append(PredictionRecord(
                    origin=origin_text, school=school, target_time=target_time,
                    horizon=horizon, cohort=cohort,
                    bucket=horizon_bucket(horizon), point=point, actual=None,
                    weather_available=weather is not None,
                    known_hours=known_hours, tier=tier, blend_weight=weight,
                    intervals=intervals, residual_log=None))
        if status is not None:
            status(f'回测 origin {index + 1}/{len(origins)} {origin_text} 完成')
    return records, model_info


def fill_actuals(book, records, eval_origin=None):
    """在评估时点补齐真实标签。

    eval_origin 默认为资料中最新的可见时点；任何 target_time 晚于该时点、
    或标签在该时点仍不可见的预测保留 actual=None，评估时跳过，绝不回填未来。
    """
    if eval_origin is None:
        cutoff = max(instant(r['known_at']) for r in book.document['records'])
    else:
        cutoff = instant(eval_origin)
    latest_snapshot = OriginSnapshot(book, cutoff.isoformat())
    for record in records:
        if instant(record.target_time) > cutoff:
            continue
        actual = latest_snapshot.load_value(record.school, record.target_time)
        if actual is None:
            continue
        record.actual = actual
        record.residual_log = log1p(actual) - log1p(record.point)
    return records


def _error_metrics(values):
    if not values:
        return None
    errors = [pred - actual for pred, actual in values]
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    actual_mean = sum(actual for _, actual in values) / len(values)
    mape_denom = sum(abs(actual) for _, actual in values)
    mape = (sum(abs_errors) / mape_denom) if mape_denom > 0 else None
    return {'n': len(values), 'mae': round(mae, 4), 'rmse': round(rmse, 4),
            'mape': round(mape, 4) if mape is not None else None,
            'mean_actual': round(actual_mean, 4),
            'bias': round(sum(errors) / len(errors), 4)}


def _coverage(values_with_intervals):
    output = {}
    for level in LEVELS:
        key = f'{level:.2f}'
        supported = [(actual, intervals[key])
                     for actual, intervals in values_with_intervals
                     if intervals.get(key) and intervals[key]['status'] != 'unsupported']
        refused = len(values_with_intervals) - len(supported)
        if not supported:
            output[key] = {'evaluated': 0, 'refused': refused, 'coverage': None,
                           'mean_width': None}
            continue
        hits = sum(1 for actual, part in supported
                   if part['lower'] <= actual <= part['upper'])
        widths = [part['upper'] - part['lower'] for _, part in supported]
        output[key] = {
            'evaluated': len(supported), 'refused': refused,
            'coverage': round(hits / len(supported), 4),
            'mean_width': round(sum(widths) / len(widths), 4),
            'nominal': float(key)}
    return output


def evaluate(records):
    """分层评估：总体、人群×距离桶、逐学校×距离桶。"""
    realized = [r for r in records if r.actual is not None]
    groups = {
        'overall': realized,
        'by_cohort_bucket': {},
        'by_school_bucket': {},
        'by_tier': {},
    }
    for record in realized:
        groups['by_cohort_bucket'].setdefault((record.cohort, record.bucket), []).append(record)
        groups['by_school_bucket'].setdefault((record.school, record.bucket), []).append(record)
        groups['by_tier'].setdefault(record.tier, []).append(record)

    def summarize(rows):
        metrics = _error_metrics([(r.point, r.actual) for r in rows])
        coverage = _coverage([(r.actual, r.intervals) for r in rows])
        return {'errors': metrics, 'coverage': coverage}

    report = {'overall': summarize(realized),
              'cohort_bucket': {}, 'school_bucket': {}, 'tier': {}}
    for (cohort, bucket), rows in sorted(groups['by_cohort_bucket'].items()):
        report['cohort_bucket'].setdefault(cohort, {})[bucket] = summarize(rows)
    for (school, bucket), rows in sorted(groups['by_school_bucket'].items()):
        report['school_bucket'].setdefault(school, {})[bucket] = summarize(rows)
    for tier, rows in sorted(groups['by_tier'].items()):
        report['tier'][tier] = summarize(rows)
    report['realized_predictions'] = len(realized)
    report['pending_predictions'] = len(records) - len(realized)
    return report
