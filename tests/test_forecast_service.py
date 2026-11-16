"""多时段预测服务的测试：时点正确性、无泄漏、冷启动、校准与工件复现。"""
import copy
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from campus_forecast.records import RecordBook, instant
from campus_forecast.dataset import OriginSnapshot
from campus_forecast.features import (FeatureBuilder, cohort_of,
                                      feature_columns, horizon_bucket)
from campus_forecast.modeling import (ForecastModel, expm1, local_weight,
                                      log1p, ridge_fit, train_model)
from campus_forecast.calibration import (LEVELS, apply_interval, calibrate,
                                         quantile)
from campus_forecast.backtest import (BacktestConfig, evaluate, fill_actuals,
                                      run_backtest)
from campus_forecast.historical import historical_calibration
from campus_forecast.service import ForecastStore, forecast_school

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'sample.json'


def iso(stamp):
    return stamp.replace(tzinfo=timezone.utc).isoformat()


def make_book():
    """构造带修订、迟到与天气双发布的合成资料，覆盖所有泄漏路径。"""
    start = datetime(2026, 10, 5, 0, 0)
    records = []
    schools = {
        'alpha': {'climate_zone': 'temperate', 'floor_area': '1000'},
        'beta': {'climate_zone': 'coastal', 'floor_area': '1200'},
        'gamma': {'climate_zone': 'temperate', 'floor_area': '900'},
    }
    for school, info in schools.items():
        records.append({'kind': 'school', 'id': school, 'revision': 1,
                        'known_at': iso(start - timedelta(days=10)), 'data': info})
    for day in range(12):
        for hour in range(24):
            target = start + timedelta(days=day, hours=hour)
            for school in schools:
                # 负荷：确定性日曲线 + 校别规模，迟到 3 小时可见，部分有修订
                base = 20.0 + 10.0 * math.sin(math.pi * (hour - 6) / 12) + (
                    10.0 if school == 'beta' else 0.0)
                value = round(max(base, 1.0), 3)
                records.append({
                    'kind': 'load', 'id': f'{school}-{target:%Y%m%d%H}',
                    'revision': 1, 'known_at': iso(target + timedelta(hours=3)),
                    'data': {'school': school,
                             'target_time': iso(target), 'kwh': str(value)}})
                if hour == 10:  # 修订版晚到，值略变
                    records.append({
                        'kind': 'load', 'id': f'{school}-{target:%Y%m%d%H}',
                        'revision': 2,
                        'known_at': iso(target + timedelta(hours=15)),
                        'data': {'school': school,
                                 'target_time': iso(target),
                                 'kwh': str(round(value + 0.5, 3))}})
                # 天气：前一天 23:00 初版、当天 02:00 修订
                early = target.replace(hour=23) - timedelta(days=1)
                records.append({
                    'kind': 'weather',
                    'id': f'{school}-{target:%Y%m%d%H}-e', 'revision': 1,
                    'known_at': iso(early + timedelta(minutes=5)),
                    'data': {'school': school, 'target_time': iso(target),
                             'issued_at': iso(early),
                             'temperature': f'{10 + 0.3 * hour:.2f}'}})
                revision_issue = target.replace(hour=2)
                records.append({
                    'kind': 'weather',
                    'id': f'{school}-{target:%Y%m%d%H}-r', 'revision': 1,
                    'known_at': iso(revision_issue + timedelta(minutes=5)),
                    'data': {'school': school, 'target_time': iso(target),
                             'issued_at': iso(revision_issue),
                             'temperature': f'{11 + 0.3 * hour:.2f}'}})
                records.append({
                    'kind': 'calendar',
                    'id': f'{school}-{target:%Y%m%d%H}', 'revision': 1,
                    'known_at': iso(target - timedelta(days=7)),
                    'data': {'school': school, 'target_time': iso(target),
                             'day_type': 'class' if 9 <= hour <= 18 else 'closed'}})
    return RecordBook({'schema': 1, 'records': records})


class SnapshotLeakageTests(unittest.TestCase):
    def setUp(self):
        self.book = make_book()

    def test_future_load_invisible(self):
        origin = iso(datetime(2026, 10, 7, 12, 0))
        snap = OriginSnapshot(self.book, origin)
        # 12:00 的负荷 15:00 才可见，13:00 的更晚
        self.assertIsNone(snap.load_value('alpha', iso(
            datetime(2026, 10, 7, 12, 0))))
        self.assertIsNone(snap.load_value('alpha', iso(
            datetime(2026, 10, 7, 13, 0))))
        # 08:00 的负荷 11:00 已可见
        self.assertIsNotNone(snap.load_value('alpha', iso(
            datetime(2026, 10, 7, 8, 0))))

    def test_revision_visibility(self):
        target = iso(datetime(2026, 10, 7, 10, 0))
        # 初版 known_at = 13:00，修订版 known_at = 次日 01:00
        early = OriginSnapshot(self.book, iso(datetime(2026, 10, 7, 20, 0)))
        late = OriginSnapshot(self.book, iso(datetime(2026, 10, 8, 2, 0)))
        self.assertIsNotNone(early.load_value('alpha', target))
        self.assertAlmostEqual(early.load_value('alpha', target),
                               late.load_value('alpha', target) - 0.5, places=3)

    def test_weather_issue_cutoff(self):
        origin = iso(datetime(2026, 10, 7, 0, 0))
        snap = OriginSnapshot(self.book, origin)
        target = iso(datetime(2026, 10, 7, 5, 0))
        row = snap.weather_for('alpha', target)
        # 00:00 时只能看到前一天 23:00 的初版，02:00 修订尚未发布
        self.assertEqual(row['data']['issued_at'],
                         iso(datetime(2026, 10, 6, 23, 0)))
        snap2 = OriginSnapshot(self.book, iso(datetime(2026, 10, 7, 3, 0)))
        row2 = snap2.weather_for('alpha', target)
        self.assertEqual(row2['data']['issued_at'],
                         iso(datetime(2026, 10, 7, 2, 0)))

    def test_backdated_future_label_rejected(self):
        # 异常回填：target 在未来但 known_at 早于 target，必须被守卫拒绝，
        # 否则未来标签会经水平估计与中性填补泄漏进训练
        start = datetime(2026, 10, 5, tzinfo=timezone.utc)
        doc = {'schema': 1, 'records': [
            {'kind': 'school', 'id': 's', 'revision': 1,
             'known_at': iso(start - timedelta(days=10)),
             'data': {'climate_zone': 'temperate', 'floor_area': '1000'}},
            {'kind': 'load', 'id': 'future', 'revision': 1,
             'known_at': iso(datetime(2026, 10, 1, tzinfo=timezone.utc)),
             'data': {'school': 's',
                      'target_time': iso(datetime(2026, 10, 10, tzinfo=timezone.utc)),
                      'kwh': '50'}}]}
        with self.assertRaises(ValueError):
            OriginSnapshot(
                RecordBook(doc),
                iso(datetime(2026, 10, 2, tzinfo=timezone.utc)))

    def test_rolling_window_excludes_target(self):
        origin = iso(datetime(2026, 10, 8, 4, 0))  # 留足电表 3 小时迟到
        snap = OriginSnapshot(self.book, origin)
        target = iso(datetime(2026, 10, 8, 0, 0))
        mean24 = snap.rolling_mean('alpha', target, 24, 6)
        prior = [snap.load_value(
            'alpha', iso(datetime(2026, 10, 7) + timedelta(hours=h)))
            for h in range(24)]
        self.assertTrue(all(v is not None for v in prior))
        self.assertAlmostEqual(mean24, sum(prior) / 24, places=6)


class ColdStartTests(unittest.TestCase):
    def setUp(self):
        self.book = make_book()

    def test_zero_history_uses_global_level(self):
        # 在任何负荷可见之前：快照本身无标签，训练必须明确拒绝而不是造假
        origin = iso(datetime(2026, 10, 4, 23, 0))
        snap = OriginSnapshot(self.book, origin)
        self.assertEqual(snap.known_hours('alpha'), 0)
        with self.assertRaises(ValueError):
            train_model([snap], 'cold', origin)
        # 用稍晚、其他学校已有历史的快照训练，新学校只能走全局水平
        later = iso(datetime(2026, 10, 5, 4, 0))
        snap_later = OriginSnapshot(self.book, later)
        model = train_model([snap_later], 'cold', later)
        _, source = model.school_level('alpha', 0)
        self.assertEqual(source, 'global')
        vector = model.builder().vector(
            snap_later, 'alpha', iso(datetime(2026, 10, 5, 12, 0)), 8)
        _, tier, weight = model.predict_log(vector, 'alpha', 0)
        self.assertEqual(tier, 'pooled-global')
        self.assertEqual(weight, 0.0)

    def test_no_future_results_borrowed(self):
        # 10-05 04:00 时只有 10-05 00:00、01:00 两个负荷可见（3 小时迟到），
        # 训练集只能包含这两个标签，不得包含之后的任何小时
        origin = iso(datetime(2026, 10, 5, 4, 0))
        snap = OriginSnapshot(self.book, origin)
        self.assertEqual(snap.known_hours('alpha'), 2)
        model = train_model([snap], 'early', origin)
        self.assertEqual(model.level_counts['alpha'], 2)
        # 10-05 05:00 的负荷 08:00 才可见，04:00 快照不得提前看到
        self.assertIsNone(snap.load_value(
            'alpha', iso(datetime(2026, 10, 5, 5, 0))))
        snap2 = OriginSnapshot(
            self.book, iso(datetime(2026, 10, 5, 9, 0)))
        self.assertIsNotNone(snap2.load_value(
            'alpha', iso(datetime(2026, 10, 5, 5, 0))))

    def test_local_weight_gates(self):
        self.assertEqual(local_weight(0), 0.0)
        self.assertEqual(local_weight(95), 0.0)
        self.assertGreater(local_weight(168), 0.0)
        self.assertLessEqual(local_weight(10_000), 0.8)


class CalibrationTests(unittest.TestCase):
    def test_quantile_matches_linear_definition(self):
        values = sorted([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(quantile(values, 0.5), 2.5)
        self.assertAlmostEqual(quantile(values, 0.0), 1.0)

    def test_fallback_chain_and_refusal(self):
        # 只给 seen/h1 充足样本，new 桶必须回退
        residuals = [('seen', 'h1', 0.1 * (i % 5 - 2)) for i in range(40)]
        table = calibrate(residuals)
        iv = apply_interval(table, 'new', 1, log1p(30.0), True)
        self.assertEqual(iv['0.95']['status'],
                         table['_global']['_global']['status'])
        self.assertIn('calibration_source', iv['0.95'])
        # 气象缺失一律拒答
        refused = apply_interval(table, 'seen', 1, log1p(30.0), False)
        self.assertEqual(refused['0.95']['status'], 'unsupported')
        self.assertIsNone(refused['0.95']['lower'])

    def test_empty_calibration_refuses(self):
        iv = apply_interval({}, 'seen', 3, log1p(30.0), True)
        self.assertEqual(iv['0.80']['status'], 'unsupported')

    def test_intervals_are_marginal_and_ordered(self):
        residuals = [('seen', 'h7_24', (i % 11 - 5) * 0.02) for i in range(60)]
        table = calibrate(residuals)
        iv = apply_interval(table, 'seen', 12, log1p(30.0), True)
        for level in ('0.80', '0.95'):
            part = iv[level]
            self.assertLess(part['lower'], part['upper'])
            self.assertGreaterEqual(part['lower'], 0.0)
        self.assertGreaterEqual(
            iv['0.95']['upper'] - iv['0.95']['lower'],
            iv['0.80']['upper'] - iv['0.80']['lower'])


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.book = make_book()

    def test_backtest_is_leak_free_and_stratified(self):
        origins = [iso(datetime(2026, 10, 9, 23, 0) + timedelta(days=d))
                   for d in range(3)]
        cfg = BacktestConfig(origins=origins, horizons=tuple(range(1, 25)))
        records, info = run_backtest(self.book, cfg)
        fill_actuals(self.book, records)
        # 每个 origin 的模型训练样本量必须严格递增且只来自当时可见资料
        counts = [item['sample_count'] for item in info]
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        report = evaluate(records)
        # 分层结构存在且不被总体平均掩盖
        self.assertIn('cohort_bucket', report)
        self.assertIn('school_bucket', report)
        for rows in report['cohort_bucket'].values():
            for summary in rows.values():
                self.assertIsNotNone(summary['errors'])
                self.assertIn('0.95', summary['coverage'])

    def test_pending_predictions_not_evaluated(self):
        # 最后一个 origin 的目标超出资料范围，必须保留为未核对而非回填
        late_origin = iso(datetime(2026, 10, 16, 23, 0))
        cfg = BacktestConfig(origins=[late_origin], horizons=tuple(range(1, 25)))
        records, _ = run_backtest(self.book, cfg)
        fill_actuals(self.book, records)
        self.assertTrue(any(r.actual is None for r in records))
        report = evaluate(records)
        self.assertGreater(report['pending_predictions'], 0)

    def test_training_series_never_contains_future_targets(self):
        # 任意快照的训练序列，所有 target_time 都必须严格早于 origin
        from campus_forecast.modeling import build_training_rows
        origin = iso(datetime(2026, 10, 12, 0, 0))
        snap = OriginSnapshot(self.book, origin)
        rows = build_training_rows([snap])
        self.assertTrue(rows)
        cutoff = instant(origin)
        for snapshot, school, target, horizon, _ in rows:
            self.assertLess(instant(target), cutoff)
            self.assertGreaterEqual(horizon, 1)
            # 训练标签必须是该 origin 快照实际可见的值
            self.assertIsNotNone(snapshot.load_value(school, target))

    def test_neutral_imputation_uses_only_visible_past(self):
        # 中性填补值只能来自严格早于目标小时的可见负荷，
        # 不含目标小时本身，也不含相对目标属于未来的小时（防目标泄漏）
        origin = iso(datetime(2026, 10, 10, 12, 0))
        snap = OriginSnapshot(self.book, origin)
        builder = FeatureBuilder(None)
        target = iso(datetime(2026, 10, 10, 12, 0))
        neutral = builder._load_neutral(snap, 'alpha', target)
        visible = [value for _, stamp, value in snap.load_series['alpha']
                   if stamp < instant(target)]
        self.assertTrue(visible)
        self.assertAlmostEqual(neutral, sum(visible) / len(visible))

    def test_neutral_imputation_excludes_own_label_and_future(self):
        # 两小时可见：00:00=100、01:00=110。为 00:00 这一行求中性值时，
        # 其自身标签 100 与"未来"的 110 都不得进入（此前的泄漏点）
        start = datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc)
        records = [{
            'kind': 'school', 'id': 's1', 'revision': 1,
            'known_at': iso(start - timedelta(days=1)),
            'data': {'climate_zone': 'temperate', 'floor_area': '1000'}}]
        for hour, value in ((0, 100.0), (1, 110.0)):
            target = start + timedelta(hours=hour)
            records.append({
                'kind': 'load', 'id': f'l{hour}', 'revision': 1,
                'known_at': iso(target + timedelta(hours=3)),
                'data': {'school': 's1', 'target_time': iso(target),
                         'kwh': str(value)}})
        book = RecordBook({'schema': 1, 'records': records})
        snap = OriginSnapshot(book, iso(datetime(2026, 10, 5, 6, 0)))
        builder = FeatureBuilder(None)
        # 目标 00:00 之前没有任何读数 -> 回退值，而不是含自身标签的 105
        neutral = builder._load_neutral(
            snap, 's1', iso(datetime(2026, 10, 5, 0, 0)))
        self.assertNotAlmostEqual(neutral, 105.0)
        # 目标 01:00 的中性值只能含 00:00 的 100，不含自身 110
        neutral2 = builder._load_neutral(
            snap, 's1', iso(datetime(2026, 10, 5, 1, 0)))
        self.assertAlmostEqual(neutral2, 100.0)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.book = make_book()
        self.origin = iso(datetime(2026, 10, 14, 23, 0))
        self.snap = OriginSnapshot(self.book, self.origin)
        self.model = train_model([self.snap], 'art', self.origin)

    def test_save_load_roundtrip_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            self.model.save(path)
            loaded = ForecastModel.load(path)
            self.assertEqual(loaded.feature_version, self.model.feature_version)
            for school in ('alpha', 'beta', 'gamma'):
                kh = self.snap.known_hours(school)
                target = iso(datetime(2026, 10, 15, 12, 0))
                v1 = self.model.builder().vector(self.snap, school, target, 13)
                v2 = loaded.builder().vector(self.snap, school, target, 13)
                self.assertEqual(v1, v2)
                p1 = self.model.predict_log(v1, school, kh)
                p2 = loaded.predict_log(v2, school, kh)
                self.assertEqual(p1, p2)

    def test_feature_version_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            self.model.save(path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['feature_version'] = 'features-999'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                ForecastModel.load(path)

    def test_column_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            self.model.save(path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['columns'] = payload['columns'][:-1]
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                ForecastModel.load(path)

    def test_store_refuses_overwrite_and_reproduces_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ForecastStore(tmp)
            store.save_model(self.model)
            with self.assertRaises(FileExistsError):
                store.save_model(self.model)
            payload = forecast_school(
                self.book, self.model, self.origin, 'beta',
                horizons=(1, 2, 12), store=store)
            reloaded = forecast_school(
                self.book, store.load_model('art'), self.origin, 'beta',
                horizons=(1, 2, 12), save=False)
            self.assertEqual(payload['forecast_id'], reloaded['forecast_id'])
            self.assertEqual(
                [r['point'] for r in payload['rows']],
                [r['point'] for r in reloaded['rows']])
            self.assertEqual(payload['rows'], reloaded['rows'])

    def test_unknown_school_refused(self):
        payload = forecast_school(
            self.book, self.model, self.origin, 'ghost',
            horizons=(1, 2), save=False)
        self.assertEqual(payload['status'], 'refused')
        self.assertEqual(payload['rows'], [])


class FeatureContractTests(unittest.TestCase):
    def test_horizon_buckets_and_cohorts(self):
        self.assertEqual(horizon_bucket(1), 'h1')
        self.assertEqual(horizon_bucket(6), 'h2_6')
        self.assertEqual(horizon_bucket(24), 'h7_24')
        self.assertEqual(horizon_bucket(40), 'h25_plus')
        self.assertEqual(horizon_bucket(999), 'h25_plus')
        self.assertEqual(cohort_of(0), 'new')
        self.assertEqual(cohort_of(168), 'new')
        self.assertEqual(cohort_of(169), 'seen')

    def test_mask_removes_unavailable_lags(self):
        book = make_book()
        origin = iso(datetime(2026, 10, 6, 23, 0))
        snap = OriginSnapshot(book, origin)
        builder = FeatureBuilder(self._stats(book))
        target = iso(datetime(2026, 10, 7, 12, 0))  # horizon 13
        unmasked, binary_unmasked = builder.raw_features(
            snap, 'alpha', target, 13, False)
        masked, binary = builder.raw_features(snap, 'alpha', target, 13, True)
        # 目标 T=origin+13：滞后 L 仅当 L>=13 时不晚于 origin。
        # lag1/lag2 指向 origin 之后（结构上不可见）-> 掩码后缺失。
        self.assertEqual(binary['lag1_missing'], 1)
        self.assertEqual(binary['lag2_missing'], 1)
        # lag24 不晚于 origin 且确有读数 -> 保持可见，不被屏蔽
        self.assertEqual(binary['lag24_missing'], 0)
        self.assertEqual(binary_unmasked['lag24_missing'], 0)
        # 未掩码时 lag1 快照查询同样为 None（未来小时），两条路径特征一致
        self.assertEqual(unmasked['lag1'], masked['lag1'])
        # 未掩码时 lag24 实际有值（目标前 24h 已发生）
        self.assertNotEqual(unmasked['lag24'], 0.0)

    @staticmethod
    def _stats(book):
        return None

    def test_ridge_recovers_linear_coefficients(self):
        # 无噪声线性关系 y = 3x1 - 2x2，第一列为免正则截距（常数 1）。
        # 两个特征列设计为正交（均值 0、内积 0），避免共线导致系数不可识别。
        rows, targets = [], []
        for i in range(-20, 21):
            x1 = 0.5 * i
            x2 = 1.0 if i % 2 == 0 else -1.0  # 与 x1 内积为 0
            rows.append([1.0, x1, x2])
            targets.append(3 * x1 - 2 * x2)
        weights = ridge_fit(rows, targets, ridge=1e-8)
        self.assertAlmostEqual(weights[0], 0.0, places=3)
        self.assertAlmostEqual(weights[1], 3.0, places=3)
        self.assertAlmostEqual(weights[2], -2.0, places=3)


class SampleDataIntegrationTests(unittest.TestCase):
    """用真实样本资料跑一次精简端到端流程。"""

    def test_sample_pipeline_and_calibration(self):
        book = RecordBook.load(SAMPLE)
        table, n = historical_calibration(
            book, '2026-11-04T00:00:00+00:00')
        self.assertGreater(n, 300)
        for cohort in ('new', 'seen'):
            self.assertIn(cohort, table)
        origins = [iso(datetime(2026, 11, 2, 0, 0) + timedelta(days=d))
                   for d in range(2)]
        records, _ = run_backtest(
            book, BacktestConfig(origins=origins, horizons=(1, 2, 12, 24)))
        fill_actuals(book, records)
        report = evaluate(records)
        self.assertGreater(report['realized_predictions'], 0)
        # 真实资料中新学校必须被识别为 new 人群
        cohorts = {r.cohort for r in records}
        self.assertIn('new', cohorts)


if __name__ == '__main__':
    unittest.main()
