"""读取资料并执行诊断或预测服务命令。

无参数时保留历史直接诊断行为；传入子命令时进入多时段预测服务。
"""
import sys

from .records import RecordBook, output
from .operations import describe


def main():
    if len(sys.argv) > 1 and not sys.argv[1].endswith('.json'):
        from .cli import main as cli_main
        cli_main()
        return
    import argparse
    parser = argparse.ArgumentParser(description='零碳校园负荷的冷启动预测与区间交付资料诊断')
    parser.add_argument('data')
    parser.add_argument('--as-of', required=True)
    args = parser.parse_args()
    print(output(describe(RecordBook.load(args.data), args.as_of)))


if __name__ == '__main__':
    main()
