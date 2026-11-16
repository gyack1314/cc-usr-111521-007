"""发布时点（point-in-time）快照与索引。

所有训练、回测与实际预测都只能看到 origin 时刻已经可见的记录：
RecordBook.rows 已按 known_at 截断，本模块在此之上建立内存索引，
并对天气额外执行 issued_at <= origin 的约束（CONTRACT.md）。
快照对象不可变，特征构建只允许读取快照，杜绝把后续修订或未来标签
悄悄带入历史样本（新学校也借不到自己的未来结果）。
"""
from datetime import timedelta

from .records import instant, number

LOAD = 'load'
WEATHER = 'weather'
CALENDAR = 'calendar'
SCHOOL = 'school'
SNAPSHOT_KINDS = (LOAD, WEATHER, CALENDAR, SCHOOL)


def hour_add(stamp, hours):
    return (instant(stamp) + timedelta(hours=hours)).isoformat()


def hour_key(stamp):
    """小时粒度的可比较键（时区统一为感知 datetime）。"""
    return instant(stamp)


class OriginSnapshot:
    """某个 origin 时刻可见资料的只读索引。"""

    def __init__(self, book, origin):
        self.book = book
        self.origin = instant(origin)
        self.origin_text = self.origin.isoformat()
        rows_by_kind = {kind: book.rows(kind, self.origin_text) for kind in SNAPSHOT_KINDS}
        # 学校静态属性（已校验 floor_area > 0）
        self.schools = {row['id']: row['data'] for row in rows_by_kind[SCHOOL]}
        # 负荷：(school, target_time) -> 该时点可见的最新修订值
        self.load_index = {}
        for row in rows_by_kind[LOAD]:
            data = row['data']
            value = number(data['kwh'])
            if value < 0:
                raise ValueError('用电量不能为负')
            # 纵深防御：电表读数只能在其目标小时之后才可见。若 known_at 不晚于
            # target_time，说明资料回填异常，放行会让未来标签泄漏进训练与预测。
            if instant(row['known_at']) <= instant(data['target_time']):
                raise ValueError(
                    '负荷记录的 known_at 必须晚于 target_time：'
                    f"{row.get('id')}")
            key = (data['school'], data['target_time'])
            old = self.load_index.get(key)
            if old is None or row['revision'] > old[0]:
                self.load_index[key] = (row['revision'], float(value))
        # 每所学校按时间排序的负荷序列，供滞后与滚动窗口使用
        self.load_series = {}
        per_school = {}
        for (school, target), (revision, value) in self.load_index.items():
            per_school.setdefault(school, []).append((target, value))
        for school, pairs in per_school.items():
            pairs.sort(key=lambda pair: instant(pair[0]))
            # 同时保存 datetime 键，避免对 ISO 字符串做字典序比较
            self.load_series[school] = [
                (stamp_text, instant(stamp_text), value)
                for stamp_text, value in pairs]
        # 天气：(school, target_time) -> 所有 origin 时可见的发布版本
        self.weather_index = {}
        for row in rows_by_kind[WEATHER]:
            data = row['data']
            if instant(data['issued_at']) > self.origin:
                # CONTRACT：issued_at 不得晚于预测 origin
                continue
            key = (data['school'], data['target_time'])
            self.weather_index.setdefault(key, []).append(row)
        # 校历：(school, target_time) -> day_type
        self.calendar_index = {}
        for row in rows_by_kind[CALENDAR]:
            data = row['data']
            self.calendar_index[(data['school'], data['target_time'])] = data['day_type']
        self.fingerprint = book.fingerprint(SNAPSHOT_KINDS, self.origin_text)

    # -- 基础查询 -------------------------------------------------------
    def school(self, school):
        info = self.schools.get(school)
        if info is None:
            raise ValueError(f'学校资料在 {self.origin_text} 不可见：{school}')
        return info

    def load_value(self, school, target_time):
        """返回该小时已知的负荷值；未知返回 None（不得与零负荷混同）。"""
        hit = self.load_index.get((school, target_time))
        return None if hit is None else hit[1]

    def weather_for(self, school, target_time):
        """选择 issued_at 不晚于 origin 的最新天气发布；无则 None。"""
        options = self.weather_index.get((school, target_time))
        if not options:
            return None
        return max(options, key=lambda r: (instant(r['data']['issued_at']), r['id']))

    def day_type(self, school, target_time):
        return self.calendar_index.get((school, target_time))

    # -- 滞后与滚动窗口 -------------------------------------------------
    def lag_value(self, school, target_time, lag_hours):
        past = hour_add(target_time, -lag_hours)
        return self.load_value(school, past)

    def rolling_mean(self, school, target_time, window_hours, min_present):
        """目标小时之前 window_hours 个完整小时的均值。

        要求窗口内至少 min_present 个已知小时，否则视为不支持（None）。
        只统计严格早于 target_time 的小时，保证不触碰目标标签本身。
        """
        series = self.load_series.get(school)
        if not series:
            return None
        target = instant(target_time)
        start = target - timedelta(hours=window_hours)
        picked = []
        for stamp_text, stamp, value in reversed(series):
            if stamp >= target:
                continue  # 严格排除目标小时本身及任何未来小时
            if stamp < start:
                break  # 窗口为闭区间 [start, target)，起点包含
            picked.append(value)
        if len(picked) < min_present:
            return None
        return sum(picked) / len(picked)

    def known_hours(self, school):
        """该学校在 origin 时可见的负荷小时数。"""
        return len(self.load_series.get(school, ()))


def training_snapshots(book, origins):
    """为多个 origin 构建快照（回测用），按 origin 去重并排序。"""
    unique = sorted({instant(o) for o in origins})
    return [OriginSnapshot(book, stamp.isoformat()) for stamp in unique]
