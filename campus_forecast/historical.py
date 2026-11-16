"""无泄漏的历史回放校准（稀疏折叠）。

走查回测回答"滚动 origin 的端到端表现"；生产模型的区间校准需要足够多、
覆盖各距离桶与人群的样本外残差。做法：

- 折叠 origin 每隔 FOLD_SPACING_DAYS 天取一个 23:00 发布时点，每个折叠
  只用该 origin 已可见的资料从零训练模型（与生产训练同一份代码路径）；
- 对折叠之后 WINDOW_HOURS 小时内、抽样小时对应的已实现标签计算样本外残差；
- 折叠模型看不到任何折叠之后的标签，残差是诚实的前向误差；
- 距离桶按真实预测距离（fold origin -> target）归入 h1/h2_6/h7_24/h25_48，
  人群按折叠 origin 时该校可见历史量归入 new/seen。
"""
from datetime import timedelta

from .calibration import calibrate
from .dataset import OriginSnapshot, instant
from .features import cohort_of, horizon_bucket
from .modeling import log1p, train_model

# 每个抽样日入选的目标小时（本地小时）：1 点保证 h1 近实时桶有样本，
# 其余覆盖凌晨/上午/午后/晚间各形态
SAMPLE_HOURS = (1, 2, 8, 12, 17, 21)
FOLD_SPACING_DAYS = 3
# 日度 23:05 发布的气象预报覆盖未来约 25 小时，回放窗口与之对齐；
# 超出气象支持的远期不进入校准池，预测时由服务层直接拒答区间。
WINDOW_HOURS = 25
MIN_FOLD_SAMPLES = 200


def _fold_origins(train_origin):
    """折叠 origin 取每日整点 00:00，与实际服务节奏对齐。

    日度气象预报在前一天 23:00 发布、23:05 入库，00:00 origin 时当天 24 个
    目标小时的预报已全部可见，因此 h1–h24 都在气象支持范围内。23:00 发布、
    23:05 入库的 5 分钟延迟意味着 23:00 origin 看不到次日预报，故服务
    不应在 23:00 运行。
    """
    end = instant(train_origin)
    origins = []
    cursor = end
    while True:
        candidate = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
        if candidate < end:
            origins.append(candidate)
        cursor = candidate - timedelta(days=FOLD_SPACING_DAYS)
        # 数据从 10-05 开始，留出至少一周的预热期
        if cursor < end - timedelta(days=30):
            break
    return sorted(origins)


def historical_residuals(book, train_origin, status=None):
    """对稀疏折叠后的目标小时做样本外回放，返回 (cohort, bucket, residual)。"""
    end = instant(train_origin)
    base_snapshot = OriginSnapshot(book, train_origin)
    residuals = []
    folds = _fold_origins(train_origin)
    for fold_index, fold in enumerate(folds):
        fold_text = fold.isoformat()
        snapshot = OriginSnapshot(book, fold_text)
        total_samples = sum(snapshot.known_hours(s) for s in snapshot.schools)
        if total_samples < MIN_FOLD_SAMPLES:
            continue
        model = train_model(
            [snapshot], f'cal-fold-{fold.strftime("%Y%m%dT%H")}', fold_text)
        builder = model.builder()
        for day in range(WINDOW_HOURS // 24 + 1):
            for hour in SAMPLE_HOURS:
                target = (fold + timedelta(days=day)).replace(
                    hour=hour, minute=0, second=0, microsecond=0)
                horizon = int((target - fold).total_seconds() // 3600)
                if horizon <= 0 or horizon > WINDOW_HOURS or target >= end:
                    continue
                target_text = target.isoformat()
                for school in sorted(snapshot.schools):
                    # 标签必须在训练 origin 已经可见，否则该残差无法核对
                    actual = base_snapshot.load_value(school, target_text)
                    if actual is None:
                        continue
                    known_hours = snapshot.known_hours(school)
                    vector = builder.vector(snapshot, school, target_text, horizon,
                                            mask_unavailable_lags=True)
                    point_log, _, _ = model.predict_log(vector, school, known_hours)
                    residual = log1p(actual) - point_log
                    residuals.append((
                        cohort_of(known_hours), horizon_bucket(horizon), residual))
        if status is not None:
            status(f'历史回放折叠 {fold_index + 1}/{len(folds)} {fold_text} 完成')
    return residuals


def historical_calibration(book, train_origin, status=None):
    """历史回放残差 -> 校准表。"""
    residuals = historical_residuals(book, train_origin, status=status)
    if not residuals:
        return {}, 0
    return calibrate(residuals), len(residuals)
