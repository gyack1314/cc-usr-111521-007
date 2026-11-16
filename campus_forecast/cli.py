"""命令行入口：训练、回测评估、预测、工件复用。"""
import argparse
import json
import sys
from datetime import timedelta

from .backtest import (BacktestConfig, DEFAULT_HORIZONS, evaluate,
                       fill_actuals, run_backtest)
from .records import instant
from .pipeline import train_production_model
from .records import RecordBook, output
from .service import ForecastStore, forecast_school


def _parse_origins(text):
    if text == 'auto':
        return None
    return [part.strip() for part in text.split(',') if part.strip()]


def _default_backtest_origins(book, origin, days):
    """从 origin 向前取每日整点 00:00 的走查时点（对齐日度服务节奏），
    首个时点保留至少一周历史。"""
    end = instant(origin)
    origins = []
    for offset in range(1, days + 1):
        candidate = (end - timedelta(days=offset)).replace(hour=0, minute=0,
                                                            second=0, microsecond=0)
        origins.append(candidate.isoformat())
    return list(reversed(origins))


def cmd_train(args):
    book = RecordBook.load(args.data)
    origins = _parse_origins(args.backtest_origins)
    if origins is None:
        origins = _default_backtest_origins(book, args.origin, args.backtest_days)
    horizons = tuple(args.horizons)
    model, records = train_production_model(
        book, args.origin, args.model_id, origins, horizons,
        status=lambda message: print(message, file=sys.stderr))
    store = ForecastStore(args.store)
    path = store.save_model(model)
    print(output({'model_id': model.model_id, 'artifact': path,
                  'trained_origin': model.trained_origin,
                  'training': model.training,
                  'calibration': model.calibration}))


def cmd_backtest(args):
    book = RecordBook.load(args.data)
    origins = _parse_origins(args.origins)
    if origins is None:
        origins = _default_backtest_origins(book, args.eval_origin, args.days)
    config = BacktestConfig(origins=origins, horizons=tuple(args.horizons))
    records, model_info = run_backtest(
        book, config, status=lambda message: print(message, file=sys.stderr))
    fill_actuals(book, records, args.eval_origin)
    report = evaluate(records)
    report['backtest_origins'] = origins
    report['models'] = model_info
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as stream:
            json.dump({'report': report,
                       'records': [r.to_dict() for r in records]},
                      stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(output(report))


def cmd_forecast(args):
    book = RecordBook.load(args.data)
    store = ForecastStore(args.store)
    model = store.load_model(args.model_id)
    payload = forecast_school(book, model, args.origin, args.school,
                              horizons=tuple(args.horizons), store=store)
    print(output(payload))


def cmd_list(args):
    store = ForecastStore(args.store)
    print(output({'models': store.list_models(),
                  'forecasts': store.list_forecasts()}))


def cmd_replay(args):
    """加载已保存预测工件并复算，验证模型版本/特征/校准复现相同预测。"""
    book = RecordBook.load(args.data)
    store = ForecastStore(args.store)
    saved = store.load_forecast(args.forecast_id)
    model = store.load_model(saved['model_id'])
    replay = forecast_school(book, model, saved['origin'], saved['school'],
                             horizons=tuple(saved['horizons']), save=False)
    same = [
        {
            'horizon': a['horizon'],
            'point_match': a['point'] == b['point'],
            'intervals_match': a['intervals'] == b['intervals'],
        }
        for a, b in zip(saved['rows'], replay['rows'])
    ]
    identical = all(item['point_match'] and item['intervals_match']
                    for item in same)
    print(output({'forecast_id': args.forecast_id,
                  'fingerprint_match': saved['data_fingerprint'] == replay['data_fingerprint'],
                  'identical': identical, 'rows': same}))


def build_parser():
    parser = argparse.ArgumentParser(description='零碳校园多时段预测服务')
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser('train', help='走查回测 + 校准 + 训练并保存模型工件')
    train.add_argument('data')
    train.add_argument('--origin', required=True)
    train.add_argument('--model-id', required=True)
    train.add_argument('--store', required=True)
    train.add_argument('--backtest-origins', default='auto')
    train.add_argument('--backtest-days', type=int, default=14)
    train.add_argument('--horizons', type=int, nargs='+', default=list(DEFAULT_HORIZONS))
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser('backtest', help='走查回测并输出分层误差与区间覆盖')
    backtest.add_argument('data')
    backtest.add_argument('--origins', default='auto')
    backtest.add_argument('--days', type=int, default=14)
    backtest.add_argument('--eval-origin')
    backtest.add_argument('--horizons', type=int, nargs='+', default=list(DEFAULT_HORIZONS))
    backtest.add_argument('--out')
    backtest.set_defaults(func=cmd_backtest)

    forecast = sub.add_parser('forecast', help='按保存的模型产出多时段预测工件')
    forecast.add_argument('data')
    forecast.add_argument('--store', required=True)
    forecast.add_argument('--model-id', required=True)
    forecast.add_argument('--origin', required=True)
    forecast.add_argument('--school', required=True)
    forecast.add_argument('--horizons', type=int, nargs='+', default=list(DEFAULT_HORIZONS))
    forecast.set_defaults(func=cmd_forecast)

    listing = sub.add_parser('list', help='列出已保存模型与预测工件')
    listing.add_argument('--store', required=True)
    listing.set_defaults(func=cmd_list)

    replay = sub.add_parser('replay', help='复算已保存预测，验证可复现性')
    replay.add_argument('data')
    replay.add_argument('--store', required=True)
    replay.add_argument('--forecast-id', required=True)
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
