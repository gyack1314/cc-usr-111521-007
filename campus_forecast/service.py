"""可保存、可复用的多时段预测服务。

职责：
- 加载已训练模型工件（含特征处理统计与校准表），对指定 origin/学校/距离集合
  产出点预测、边际预测区间、适用范围与降级/拒答说明；
- ForecastStore 以不可变工件形式保存每次预测与每个模型，相同输入必然得到
  相同输出，工件可被后续运营流程复用而不重新训练；
- 资料支持不足时显式降级或拒答区间，绝不输出无法负责的"精确数字"。
"""
import hashlib
import json
import os
from dataclasses import dataclass

from .backtest import DEFAULT_HORIZONS
from .calibration import LEVELS, apply_interval
from .dataset import OriginSnapshot, hour_add
from .features import cohort_of, horizon_bucket
from .modeling import ForecastModel, expm1
from .records import instant

ARTIFACT_SCHEMA = 1
INTERVAL_KIND = 'marginal'  # 边际（逐点）覆盖，非联合覆盖


class ForecastStore:
    """模型与预测工件的目录存储，全部为 JSON、写入后不可变。"""

    def __init__(self, root):
        self.root = root
        self.models_dir = os.path.join(root, 'models')
        self.forecasts_dir = os.path.join(root, 'forecasts')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.forecasts_dir, exist_ok=True)

    def save_model(self, model):
        path = os.path.join(self.models_dir, f'{model.model_id}.json')
        if os.path.exists(path):
            raise FileExistsError(f'模型工件已存在，拒绝覆盖：{path}')
        model.save(path)
        return path

    def load_model(self, model_id):
        return ForecastModel.load(os.path.join(self.models_dir, f'{model_id}.json'))

    def list_models(self):
        return sorted(name[:-5] for name in os.listdir(self.models_dir)
                      if name.endswith('.json'))

    def save_forecast(self, payload):
        forecast_id = payload['forecast_id']
        path = os.path.join(self.forecasts_dir, f'{forecast_id}.json')
        if os.path.exists(path):
            raise FileExistsError(f'预测工件已存在，拒绝覆盖：{path}')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
        return path

    def load_forecast(self, forecast_id):
        with open(os.path.join(self.forecasts_dir, f'{forecast_id}.json'),
                  encoding='utf-8') as stream:
            return json.load(stream)

    def list_forecasts(self):
        return sorted(name[:-5] for name in os.listdir(self.forecasts_dir)
                      if name.endswith('.json'))


def _forecast_id(model_id, origin, school, horizons, fingerprint):
    payload = json.dumps(
        [model_id, origin, school, sorted(horizons), fingerprint],
        ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ForecastRow:
    horizon: int
    target_time: str
    point: float
    tier: str
    blend_weight: float
    intervals: dict
    status: str
    reasons: list
    support: dict


def forecast_school(book, model, origin, school, horizons=DEFAULT_HORIZONS,
                    store=None, save=True):
    """对单个学校执行多时段预测并（可选）保存工件。

    每一行都带 status：
      - ok：点预测与至少一个校准区间可用；
      - degraded：点预测可用，但部分/全部区间因资料不足拒答或仅稳健校准；
      - refused：学校在 origin 不可见等硬失败，整体不产出。
    """
    origin_text = instant(origin).isoformat()
    horizons = tuple(horizons)
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons) \
            or len(set(horizons)) != len(horizons):
        raise ValueError('预测距离须为互不相同的正整数小时')
    snapshot = OriginSnapshot(book, origin_text)
    if school not in snapshot.schools:
        return {'status': 'refused',
                'reasons': [f'学校 {school} 在 {origin_text} 没有可见静态资料'],
                'rows': []}
    builder = model.builder()
    known_hours = snapshot.known_hours(school)
    cohort = cohort_of(known_hours)
    # 动态计算该 origin 下该校天气实际支持的最大预测距离，
    # 而不是写死假设：逐日检查请求距离内的目标小时是否有可见预报。
    weather_supported_horizons = [
        h for h in sorted(horizons)
        if snapshot.weather_for(school, hour_add(origin_text, h)) is not None]
    weather_limit = max(weather_supported_horizons) if weather_supported_horizons else 0
    rows = []
    warnings = set()
    for horizon in sorted(horizons):
        target_time = hour_add(origin_text, horizon)
        support = builder.support_flags(snapshot, school, target_time, horizon)
        vector = builder.vector(snapshot, school, target_time, horizon)
        point_log, tier, weight = model.predict_log(vector, school, known_hours)
        point = round(expm1(point_log), 4)
        intervals = apply_interval(
            model.calibration, cohort, horizon, point_log,
            weather_available=support['weather_available'])
        reasons = []
        row_status = 'ok'
        if not support['weather_available']:
            reasons.append('目标小时缺少不晚于 origin 的气象预报，区间拒答')
        if support['missing_lags']:
            reasons.append(
                '滞后特征缺失：' + ','.join(f'lag{lag}' for lag in support['missing_lags']))
        for level in LEVELS:
            part = intervals[f'{level:.2f}']
            if part['status'] == 'unsupported':
                row_status = 'degraded'
                warnings.add(f'{level:.0%} 区间在 {horizon_bucket(horizon)} 桶不可用')
            elif part['status'] == 'robust':
                row_status = 'degraded'
                warnings.add(f'{level:.0%} 区间仅稳健校准（n={part["calibration_n"]}）')
        if cohort == 'new':
            reasons.append('新学校人群：仅使用全局冷启动模型，未借用任何本校未来结果')
        rows.append(ForecastRow(
            horizon=horizon, target_time=target_time, point=point, tier=tier,
            blend_weight=round(weight, 4), intervals=intervals,
            status=row_status, reasons=reasons, support=support))
    overall_status = 'ok'
    if any(row.status == 'degraded' for row in rows):
        overall_status = 'degraded'
    forecast_id = _forecast_id(
        model.model_id, origin_text, school, sorted(horizons), snapshot.fingerprint)
    payload = {
        'schema': ARTIFACT_SCHEMA,
        'forecast_id': forecast_id,
        'model_id': model.model_id,
        'feature_version': model.feature_version,
        'method': model.method,
        'origin': origin_text,
        'school': school,
        'horizons': sorted(horizons),
        'interval_kind': INTERVAL_KIND,
        'calibration_levels': [f'{level:.2f}' for level in LEVELS],
        'applicability': {
            'cohort': cohort,
            'known_hours': known_hours,
            'calibrated_buckets': {
                cohort_name: sorted(
                    bucket for bucket, entry in groups.items()
                    if not bucket.startswith('_') and entry['status'] != 'unsupported')
                for cohort_name, groups in model.calibration.items()
                if not cohort_name.startswith('_')},
            'weather_horizon_limit_hours': weather_limit,
            'weather_missing_horizons': [
                h for h in sorted(horizons) if h not in weather_supported_horizons],
            'notes': ['区间为逐点边际覆盖，不保证整段路径联合覆盖',
                      '点预测在气象缺失时仍可产出，但对应区间拒答',
                      '桶内样本不足时按 同人群汇总 -> 全局 顺序回退并标注来源']},
        'data_fingerprint': snapshot.fingerprint,
        'status': overall_status,
        'warnings': sorted(warnings),
        'rows': [row.__dict__ for row in rows],
    }
    if store is not None and save:
        # 工件路径是存储位置而非模型/资料内容，不写入工件本体，
        # 保证保存前后的 forecast_id 与逐行预测完全一致、可复算。
        store.save_forecast(payload)
    return payload
