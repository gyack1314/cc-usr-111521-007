"""预测区间的经验分位数校准与分层回退。

残差在 log1p(kWh) 空间按「人群（new/seen）× 距离桶」收集，区间为边际
（逐点）覆盖而非联合覆盖，结果中明确声明（CONTRACT.md）。

校准查找按支持度分层回退，并在每条区间上标注来源与状态：
  1. 该人群 × 该距离桶样本充足：直接经验分位数（calibrated）；
  2. 样本偏少：中位数 + MAD 稳健构造并按小样本 t 倍数加宽（robust）；
  3. 该桶没有样本：回退到同人群所有桶的汇总池（fallback-cohort）；
  4. 人群池也没有：回退到跨人群全局池（fallback-global），同时提示
     新学校借用的是已见学校的离散度，运营方应据此降低信任；
  5. 连全局池都不足或气象缺失：拒答区间（unsupported），点预测仍可给出。
"""
import math

from .features import horizon_bucket

LEVELS = (0.80, 0.95)
MIN_CALIBRATION_SAMPLES = 20
MIN_ROBUST_SAMPLES = 8
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
         6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         15: 2.131, 20: 2.086, 30: 2.042, 1000: 1.962}
_Z = {0.80: 1.2816, 0.95: 1.96}
_GLOBAL_COHORT = '_global'


def quantile(sorted_values, probability):
    """线性插值分位数（与 numpy 默认 linear 方法一致）。"""
    if not sorted_values:
        raise ValueError('空样本无法计算分位数')
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _t_multiplier(n):
    if n in _T_95:
        return _T_95[n] / 1.96
    nearest = min(_T_95, key=lambda k: abs(k - (n - 1)))
    return _T_95[nearest] / 1.96


def _sample_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1))


def _entry_from_values(values):
    """把一个残差池压缩成各置信水平的上下半距与状态。"""
    values = sorted(values)
    n = len(values)
    entry = {'n': n}
    if n >= MIN_CALIBRATION_SAMPLES:
        entry['status'] = 'calibrated'
        for level in LEVELS:
            alpha = (1 - level) / 2
            entry[f'{level:.2f}'] = {
                'lower': quantile(values, alpha),
                'upper': quantile(values, 1 - alpha)}
    elif n >= MIN_ROBUST_SAMPLES:
        entry['status'] = 'robust'
        median = quantile(values, 0.5)
        deviations = sorted(abs(v - median) for v in values)
        mad = quantile(deviations, 0.5) or _sample_std(values)
        scale = mad * 1.4826
        multiplier = _t_multiplier(n)
        for level in LEVELS:
            z = _Z[level]
            entry[f'{level:.2f}'] = {
                'lower': median - z * multiplier * scale,
                'upper': median + z * multiplier * scale}
    else:
        entry['status'] = 'unsupported'
        for level in LEVELS:
            entry[f'{level:.2f}'] = None
    return entry


def calibrate(residuals):
    """构造校准表：人群×距离桶 + 人群汇总池 + 全局汇总池。

    residuals: 可迭代的 (cohort, bucket, residual_log)。
    """
    pools = {}
    cohort_pools = {}
    global_values = []
    for cohort, bucket, residual in residuals:
        pools.setdefault((cohort, bucket), []).append(residual)
        cohort_pools.setdefault(cohort, []).append(residual)
        global_values.append(residual)
    table = {}
    for (cohort, bucket), values in sorted(pools.items()):
        table.setdefault(cohort, {})[bucket] = _entry_from_values(values)
    for cohort, values in sorted(cohort_pools.items()):
        table.setdefault(cohort, {})['_cohort'] = _entry_from_values(values)
    table[_GLOBAL_COHORT] = {'_global': _entry_from_values(global_values)}
    return table


def _resolve_entry(calibration, cohort, bucket):
    """按 组 -> 人群汇总 -> 全局 的顺序找可用校准池，返回 (entry, source)。"""
    group = calibration.get(cohort, {})
    entry = group.get(bucket)
    if entry is not None and entry['status'] != 'unsupported':
        return entry, f'{cohort}/{bucket}'
    cohort_entry = group.get('_cohort')
    if cohort_entry is not None and cohort_entry['status'] != 'unsupported':
        return cohort_entry, f'{cohort}/_cohort'
    global_entry = calibration.get(_GLOBAL_COHORT, {}).get('_global')
    if global_entry is not None and global_entry['status'] != 'unsupported':
        return global_entry, f'{_GLOBAL_COHORT}/_global'
    return None, None


def apply_interval(calibration, cohort, horizon, point_log, weather_available):
    """给出点预测对应的 kWh 空间区间、状态与校准来源。"""
    from .modeling import expm1
    bucket = horizon_bucket(horizon)
    result = {}
    if not weather_available:
        for level in LEVELS:
            result[f'{level:.2f}'] = {
                'lower': None, 'upper': None, 'status': 'unsupported',
                'reason': '目标小时缺少不晚于 origin 的气象预报，区间不可依赖'}
        return result
    entry, source = _resolve_entry(calibration, cohort, bucket)
    if entry is None:
        for level in LEVELS:
            result[f'{level:.2f}'] = {
                'lower': None, 'upper': None, 'status': 'unsupported',
                'reason': '所有校准池（含全局回退）样本均不足'}
        return result
    fallback_note = None
    if not source.startswith(f'{cohort}/{bucket}'):
        fallback_note = f'本桶样本不足，区间借用校准池 {source}'
        if cohort == 'new' and source.startswith(_GLOBAL_COHORT):
            fallback_note += '（新学校借用已见学校离散度，请降低信任）'
    for level in LEVELS:
        key = f'{level:.2f}'
        part = entry[key]
        if part is None:
            result[key] = {'lower': None, 'upper': None, 'status': 'unsupported',
                           'reason': f'校准池 {source} 样本不足'}
            continue
        result[key] = {
            'lower': round(expm1(point_log + part['lower']), 4),
            'upper': round(expm1(point_log + part['upper']), 4),
            'status': entry['status'],
            'calibration_n': entry['n'],
            'calibration_source': source}
        if fallback_note:
            result[key]['reason'] = fallback_note
    return result
