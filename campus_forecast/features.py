"""特征工程：只依赖快照中已可见的信息。

同一份特征定义同时用于训练、回测与实际预测，保证模型工件加载后
得到相同预测。缺失的滞后/滚动/天气特征以缺失指示位 + 中性值表示，
绝不把"未知"填成 0 负荷（CONTRACT：未知与零负荷不能混同）。
"""
import math
from dataclasses import dataclass, field

from .dataset import hour_add, instant
from .records import number

TAU = 2 * math.pi

# 滞后小时与滚动窗口（小时，所需最少已知点数）
LAGS = (1, 2, 24, 168)
ROLLING_WINDOWS = ((24, 6), (168, 24))
DAY_TYPES = ('class', 'closed')
ZONES = ('temperate', 'coastal')
# 区间校准按预测距离分桶，与日度 23:05 发布、25 小时气象覆盖的运营节奏对齐：
# h1 近实时、h2_6 次日凌晨/清晨、h7_24 次日全天（主交付时段）、h25_plus 远期。
# 走查每天滚动一次，每个桶每天都能收到新残差，避免桶粒度过细
# 导致校准池长期样本不足。
HORIZON_BUCKETS = ((1, 'h1'), (6, 'h2_6'), (24, 'h7_24'), (10_000, 'h25_plus'))
# 目标采用 log1p 的 kWh 阈值（与建模一致）
MIN_SAMPLES_LOCAL = 96
COLD_HOURS = 168  # 可见历史不超过该值视为新学校人群


def horizon_bucket(horizon):
    """返回该预测距离所属的校准/评估桶。"""
    name = 'extended'
    for threshold, bucket in HORIZON_BUCKETS:
        if horizon <= threshold:
            name = bucket
            break
    return name


def cohort_of(known_hours):
    """按 origin 时可见历史划分人群：新学校 / 已见学校。"""
    return 'new' if known_hours <= COLD_HOURS else 'seen'


# 连续特征列（标准化对象）；缺失指示位与独热列保持 0/1 不标准化
def continuous_columns():
    cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'temp_z',
            'log_area', 'horizon']
    for lag in LAGS:
        cols.append(f'lag{lag}')
    for window, _ in ROLLING_WINDOWS:
        cols.append(f'roll{window}')
    return cols


def binary_columns():
    cols = []
    for day_type in DAY_TYPES:
        cols.append(f'day_{day_type}')
    for zone in ZONES:
        cols.append(f'zone_{zone}')
    for bucket in ('h1', 'h2_6', 'h7_24', 'h25_plus'):
        cols.append(f'hb_{bucket}')
    cols.append('temp_missing')
    for lag in LAGS:
        cols.append(f'lag{lag}_missing')
    for window, _ in ROLLING_WINDOWS:
        cols.append(f'roll{window}_missing')
    return cols


def feature_columns():
    # intercept 为常数 1，使岭回归在标准化空间有显式截距，消除系统偏差
    return ['intercept'] + continuous_columns() + binary_columns()


@dataclass
class FeatureStats:
    """训练集统计量：连续列均值/标准差，以及气温的全局/校级标准化参数。"""
    means: dict
    stds: dict
    temp_global: tuple
    temp_by_school: dict = field(default_factory=dict)

    def to_dict(self):
        return {'means': self.means, 'stds': self.stds,
                'temp_global': list(self.temp_global),
                'temp_by_school': {k: list(v) for k, v in self.temp_by_school.items()}}

    @classmethod
    def from_dict(cls, payload):
        return cls(means=dict(payload['means']), stds=dict(payload['stds']),
                   temp_global=tuple(payload['temp_global']),
                   temp_by_school={k: tuple(v) for k, v in payload['temp_by_school'].items()})


class FeatureBuilder:
    """把快照 + 目标小时组装成固定顺序的特征向量。"""

    def __init__(self, stats=None):
        self.stats = stats
        self.columns = feature_columns()

    def _temp_params(self, school):
        if self.stats is not None and school in self.stats.temp_by_school:
            return self.stats.temp_by_school[school]
        if self.stats is not None:
            return self.stats.temp_global
        return (0.0, 1.0)

    def _load_neutral(self, snapshot, school, target_time):
        """滞后/滚动特征缺失时的中性填补值。

        只允许使用严格早于 target_time 的已可见负荷，绝不包含目标小时本身
        或相对目标属于"未来"的小时——否则训练行的缺失特征会掺入该行自己的
        标签，造成目标泄漏。尚无更早读数时回退到训练集 lag1 的均值。
        这样"未知"不会被误当成零负荷（CONTRACT 要求未知与零负荷区分）。
        """
        target = instant(target_time)
        series = snapshot.load_series.get(school)
        if series:
            past = [value for _, stamp, value in series if stamp < target]
            if past:
                return sum(past) / len(past)
        if self.stats is not None:
            return self.stats.means.get('lag1', 0.0)
        return 0.0

    def raw_features(self, snapshot, school, target_time, horizon,
                     mask_unavailable_lags=True):
        """返回未标准化的连续特征与 0/1 特征（顺序与 feature_columns 对齐）。

        mask_unavailable_lags=True 时，把在 horizon 距离下结构上不可见的
        滞后/滚动特征当作缺失（与真实预测条件一致）。目标 T=origin+h 的
        滞后 L 读取 T-L，仅当 L>=h 时该时刻不晚于 origin；因此 L<h 的滞后
        处于未来，必须屏蔽。滚动窗口 W 同理，仅当 W>=h 时整体可见。
        """
        info = snapshot.school(school)
        target = instant(target_time)
        hour = target.hour
        dow = target.weekday()
        day_type = snapshot.day_type(school, target_time) or 'closed'
        zone = info['climate_zone']
        area = float(number(info['floor_area']))

        weather = snapshot.weather_for(school, target_time)
        temp_missing = weather is None
        if temp_missing:
            temp_z = 0.0
        else:
            temp = float(number(weather['data']['temperature']))
            mean, std = self._temp_params(school)
            temp_z = (temp - mean) / std if std > 0 else 0.0

        cont = {
            'hour_sin': math.sin(TAU * hour / 24),
            'hour_cos': math.cos(TAU * hour / 24),
            'dow_sin': math.sin(TAU * dow / 7),
            'dow_cos': math.cos(TAU * dow / 7),
            'temp_z': temp_z,
            'log_area': math.log(area),
            'horizon': float(horizon),
        }
        binary = {}
        neutral = self._load_neutral(snapshot, school, target_time)
        for lag in LAGS:
            value = snapshot.load_value(school, hour_add(target_time, -lag))
            if mask_unavailable_lags and lag < horizon:
                value = None  # 该滞后指向 origin 之后，结构上不可见
            name = f'lag{lag}'
            if value is None:
                cont[name] = neutral
                binary[f'{name}_missing'] = 1
            else:
                cont[name] = value
                binary[f'{name}_missing'] = 0
        for window, min_present in ROLLING_WINDOWS:
            value = snapshot.rolling_mean(school, target_time, window, min_present)
            if mask_unavailable_lags and window < horizon:
                value = None  # 窗口起点晚于 origin，整体处于未来
            name = f'roll{window}'
            if value is None:
                cont[name] = neutral
                binary[f'{name}_missing'] = 1
            else:
                cont[name] = value
                binary[f'{name}_missing'] = 0

        for candidate in DAY_TYPES:
            binary[f'day_{candidate}'] = 1 if day_type == candidate else 0
        for candidate in ZONES:
            binary[f'zone_{candidate}'] = 1 if zone == candidate else 0
        bucket = horizon_bucket(horizon)
        for candidate in ('h1', 'h2_6', 'h7_24', 'h25_plus'):
            binary[f'hb_{candidate}'] = 1 if bucket == candidate else 0
        binary['temp_missing'] = 1 if temp_missing else 0
        return cont, binary

    def vector(self, snapshot, school, target_time, horizon,
               mask_unavailable_lags=True):
        """标准化后的完整特征向量（固定列顺序）。

        默认启用结构不可见滞后的屏蔽，使训练、回测与实际预测看到完全相同
        的缺失条件；被屏蔽的滞后本来在快照中也查不到值，因此与自然查询一致。
        """
        cont, binary = self.raw_features(
            snapshot, school, target_time, horizon,
            mask_unavailable_lags=mask_unavailable_lags)
        values = [1.0]  # 截距项
        for col in self.columns[1:]:
            if col in cont:
                value = cont[col]
                if self.stats is not None:
                    mean = self.stats.means.get(col, 0.0)
                    std = self.stats.stds.get(col, 1.0)
                    value = (value - mean) / std if std > 0 else 0.0
                values.append(value)
            else:
                values.append(float(binary[col]))
        return values

    def support_flags(self, snapshot, school, target_time, horizon):
        """描述该预测点的资料支持情况，供降级/拒答使用。"""
        weather = snapshot.weather_for(school, target_time)
        flags = {'weather_available': weather is not None,
                 'known_hours': snapshot.known_hours(school),
                 'cohort': cohort_of(snapshot.known_hours(school)),
                 'horizon_bucket': horizon_bucket(horizon)}
        missing_lags = []
        for lag in LAGS:
            if snapshot.load_value(school, hour_add(target_time, -lag)) is None:
                missing_lags.append(lag)
        flags['missing_lags'] = missing_lags
        return flags


def fit_stats(snapshots, samples):
    """根据训练样本估计标准化统计量。

    samples 为 (snapshot, school, target_time, horizon) 列表；
    气温统计分学校估计，样本不足的学校回退到全局统计。
    """
    builder = FeatureBuilder()
    sums = {col: 0.0 for col in continuous_columns()}
    sqs = {col: 0.0 for col in continuous_columns()}
    count = 0
    temp_by_school_values = {}
    all_temps = []
    for snapshot, school, target, horizon in samples:
        cont, binary = builder.raw_features(
            snapshot, school, target, horizon, mask_unavailable_lags=True)
        for col, value in cont.items():
            sums[col] += value
            sqs[col] += value * value
        count += 1
        weather = snapshot.weather_for(school, target)
        if weather is not None:
            temp = float(number(weather['data']['temperature']))
            temp_by_school_values.setdefault(school, []).append(temp)
            all_temps.append(temp)
    if count == 0:
        raise ValueError('没有可用于拟合特征统计的样本')
    means, stds = {}, {}
    for col in sums:
        mean = sums[col] / count
        variance = max(sqs[col] / count - mean * mean, 0.0)
        means[col] = mean
        stds[col] = variance ** 0.5
    global_mean = sum(all_temps) / len(all_temps)
    global_var = sum((t - global_mean) ** 2 for t in all_temps) / max(len(all_temps), 1)
    temp_global = (global_mean, global_var ** 0.5 or 1.0)
    temp_by_school = {}
    for school, values in temp_by_school_values.items():
        if len(values) < 24:
            continue
        mean = sum(values) / len(values)
        var = sum((t - mean) ** 2 for t in values) / len(values)
        temp_by_school[school] = (mean, var ** 0.5 or 1.0)
    return FeatureStats(means=means, stds=stds, temp_global=temp_global,
                        temp_by_school=temp_by_school)
