"""纯标准库岭回归与层次化冷启动模型。

模型结构（全部在 log1p(kWh) 空间）：
1. 校级水平 level_s：该校已可见标签均值。用经验贝叶斯方式向全局水平做
   部分池化——可见历史越少，收缩越强；没有任何历史时回退全局水平。
   新学校只要有少量已发生的读数就能获得自己的规模估计，且完全不触碰未来。
2. 全局形态模型 pooled：在所有学校的「去水平残差」上拟合岭回归，学习
   小时曲线、课型、气温与距离效应，是所有学校共享、可复现的形态先验。
3. 校级残差模型 local：仅当本校样本达到 MIN_SAMPLES_LOCAL 时拟合，
   预测时按可见历史量与全局形态收缩混合，避免短历史学校得到伪精确结果。

点预测 = level_s + 混合后的形态残差，再 expm1 还原并截断到 0。
只用 Python 正规方程 + Cholesky 分解，工件保存/加载后预测逐位一致。
"""
import json
import math
from dataclasses import dataclass, field

from .features import (COLD_HOURS, FeatureBuilder, FeatureStats,
                       MIN_SAMPLES_LOCAL, feature_columns, fit_stats)
from .records import instant

RIDGE_LAMBDA = 1.0
TARGET_FLOOR = 0.0
# 部分池化强度：达到该有效样本量时收缩比例降到约 1/2 量级。
# 取较小值，使新学校 1–2 天的读数就能主导自己的规模估计，
# 同时仍保留少量向全局水平的收缩以抵御首日时段不全造成的选择偏差。
POOLING_STRENGTH = 8.0
# 残差混合中，达到 COLD_HOURS 可见小时后校级模型的最大权重
LOCAL_MAX_WEIGHT = 0.8


def log1p(value):
    return math.log1p(max(value, 0.0))


def expm1(value):
    return max(math.expm1(value), TARGET_FLOOR)


def _cholesky(matrix):
    """对称正定矩阵的 Cholesky 分解，返回下三角 L（A = L L^T）。"""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                diag = matrix[i][i] - total
                if diag <= 1e-12:
                    raise ValueError('正规方程矩阵非正定，检查岭参数')
                lower[i][j] = math.sqrt(diag)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _solve_lower(lower, vector):
    n = len(lower)
    out = [0.0] * n
    for i in range(n):
        out[i] = (vector[i] - sum(lower[i][k] * out[k] for k in range(i))) / lower[i][i]
    return out


def _solve_upper(upper, vector):
    n = len(upper)
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (vector[i] - sum(upper[k][i] * out[k]
                                  for k in range(i + 1, n))) / upper[i][i]
    return out


def ridge_fit(matrix_x, vector_y, ridge=RIDGE_LAMBDA):
    """正规方程 (X^T X + lambda I*) w = X^T y，返回系数向量。

    I* 对截距列（第一列）取 0，其余特征取 ridge，保证截距不被收缩。
    """
    n_features = len(matrix_x[0])
    gram = [[0.0] * n_features for _ in range(n_features)]
    cross = [0.0] * n_features
    for row, target in zip(matrix_x, vector_y):
        for i, xi in enumerate(row):
            cross[i] += xi * target
            for j in range(i, n_features):
                gram[i][j] += xi * row[j]
    for i in range(1, n_features):
        gram[i][i] += ridge
        for j in range(i):
            gram[i][j] = gram[j][i]
    lower = _cholesky(gram)
    intermediate = _solve_lower(lower, cross)
    return _solve_upper(lower, intermediate)


def dot(weights, vector):
    return sum(w * x for w, x in zip(weights, vector))


def local_weight(known_hours):
    """校级残差模型的收缩权重，历史不足时为 0，封顶 LOCAL_MAX_WEIGHT。"""
    if known_hours < MIN_SAMPLES_LOCAL:
        return 0.0
    progress = (known_hours - MIN_SAMPLES_LOCAL) / max(
        COLD_HOURS - MIN_SAMPLES_LOCAL, 1)
    return LOCAL_MAX_WEIGHT * min(max(progress, 0.0), 1.0)


@dataclass
class RidgeModel:
    weights: list
    stats: FeatureStats

    def predict(self, vector):
        return dot(self.weights, vector)

    def to_dict(self):
        return {'weights': self.weights, 'stats': self.stats.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        return cls(weights=list(payload['weights']),
                   stats=FeatureStats.from_dict(payload['stats']))


@dataclass
class ForecastModel:
    """可持久化的完整模型工件：特征列、水平、形态模型、校级模型、校准状态。"""
    schema: int
    model_id: str
    trained_origin: str
    columns: list
    pooled: RidgeModel
    global_level: float
    levels: dict
    level_counts: dict
    locals: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    feature_version: str = 'features-7'
    method: str = 'hierarchical-partial-pooling-ridge-log1p'

    def builder(self):
        return FeatureBuilder(self.pooled.stats)

    def school_level(self, school, known_hours):
        """经验贝叶斯部分池化后的校级水平（log1p 空间）。"""
        if school not in self.levels or known_hours <= 0:
            return self.global_level, 'global'
        school_level = self.levels[school]
        n = self.level_counts[school]
        shrink = n / (n + POOLING_STRENGTH)
        pooled_level = shrink * school_level + (1 - shrink) * self.global_level
        return pooled_level, 'shrunk' if shrink < 0.999 else 'school'

    def predict_log(self, vector, school, known_hours):
        """返回 log1p 空间点预测、使用层级与混合权重。"""
        level, level_source = self.school_level(school, known_hours)
        pooled_shape = self.pooled.predict(vector)
        weight = local_weight(known_hours)
        local = self.locals.get(school)
        if weight <= 0.0 or local is None:
            return level + pooled_shape, f'pooled-{level_source}', 0.0
        local_shape = local.predict(vector)
        shape = (1.0 - weight) * pooled_shape + weight * local_shape
        return level + shape, f'blended-{level_source}', weight

    def to_dict(self):
        return {
            'schema': self.schema,
            'model_id': self.model_id,
            'feature_version': self.feature_version,
            'method': self.method,
            'trained_origin': self.trained_origin,
            'columns': self.columns,
            'global_level': self.global_level,
            'levels': self.levels,
            'level_counts': self.level_counts,
            'pooled': self.pooled.to_dict(),
            'locals': {school: model.to_dict() for school, model in self.locals.items()},
            'calibration': self.calibration,
            'training': self.training,
            'pooling': {'strength': POOLING_STRENGTH,
                        'local_max_weight': LOCAL_MAX_WEIGHT,
                        'min_samples_local': MIN_SAMPLES_LOCAL,
                        'cold_hours': COLD_HOURS},
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get('schema') != 1 or payload.get('feature_version') != 'features-7':
            raise ValueError('不支持的模型工件版本')
        columns = payload['columns']
        if columns != feature_columns():
            raise ValueError('特征列与当前特征处理不一致，拒绝加载该工件')
        locals_ = {school: RidgeModel.from_dict(part)
                   for school, part in payload.get('locals', {}).items()}
        return cls(
            schema=payload['schema'], model_id=payload['model_id'],
            trained_origin=payload['trained_origin'], columns=columns,
            pooled=RidgeModel.from_dict(payload['pooled']),
            global_level=payload['global_level'],
            levels=dict(payload['levels']),
            level_counts=dict(payload['level_counts']),
            locals=locals_, calibration=payload.get('calibration', {}),
            training=payload.get('training', {}),
            feature_version=payload['feature_version'],
            method=payload.get('method', 'hierarchical-partial-pooling-ridge-log1p'))

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            return cls.from_dict(json.load(stream))


def build_training_rows(snapshots):
    """从快照构造训练样本，严格保证标签在样本 origin 时已经可见。"""
    rows = []
    for snapshot in snapshots:
        for school, series in snapshot.load_series.items():
            for target_time, _, value in series:
                lag_hours = int(
                    (snapshot.origin - instant(target_time)).total_seconds() // 3600)
                horizon = max(lag_hours, 1)
                rows.append((snapshot, school, target_time, horizon, log1p(value)))
    return rows


def train_model(snapshots, model_id, trained_origin, calibration=None,
                ridge=RIDGE_LAMBDA):
    """在给定快照集合上拟合水平、全局形态模型与满足样本量的校级模型。"""
    rows = build_training_rows(snapshots)
    if not rows:
        raise ValueError('训练快照中没有任何可见标签，拒绝训练')
    sample_keys = [(snapshot, school, target, horizon)
                   for snapshot, school, target, horizon, _ in rows]
    stats = fit_stats(snapshots, sample_keys)
    builder = FeatureBuilder(stats)

    # 1) 校级水平与全局水平（仅使用各自可见标签）
    level_sum = {}
    level_n = {}
    for _, school, _, _, target_log in rows:
        level_sum[school] = level_sum.get(school, 0.0) + target_log
        level_n[school] = level_n.get(school, 0) + 1
    levels = {school: level_sum[school] / level_n[school] for school in level_sum}
    global_level = sum(level_sum.values()) / sum(level_n.values())

    # 2) 全局形态模型：目标为去水平残差；训练时按 horizon 屏蔽不可见滞后，
    #    使训练样本的缺失条件与真实多时段预测一致，消除自回归系数导致的偏差
    matrix_x, vector_y = [], []
    per_school_x = {}
    per_school_y = {}
    for snapshot, school, target, horizon, target_log in rows:
        vector = builder.vector(snapshot, school, target, horizon,
                                mask_unavailable_lags=True)
        residual = target_log - levels[school]
        matrix_x.append(vector)
        vector_y.append(residual)
        per_school_x.setdefault(school, []).append(vector)
        per_school_y.setdefault(school, []).append(residual)
    pooled = RidgeModel(ridge_fit(matrix_x, vector_y, ridge), stats)

    # 3) 校级残差模型
    locals_ = {}
    local_counts = {}
    for school, school_x in per_school_x.items():
        if len(school_x) >= MIN_SAMPLES_LOCAL:
            locals_[school] = RidgeModel(
                ridge_fit(school_x, per_school_y[school], ridge), stats)
            local_counts[school] = len(school_x)
    return ForecastModel(
        schema=1, model_id=model_id, trained_origin=trained_origin,
        columns=feature_columns(), pooled=pooled,
        global_level=global_level, levels=levels, level_counts=level_n,
        locals=locals_, calibration=calibration or {},
        training={'sample_count': len(rows),
                  'schools': sorted(level_n),
                  'local_sample_counts': local_counts,
                  'ridge_lambda': ridge,
                  'target_transform': 'log1p',
                  'shape_target': 'within_school_residual'})
