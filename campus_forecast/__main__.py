"""读取资料并执行现有直接诊断。"""
import argparse
from .records import RecordBook,output
from .operations import describe

def main():
    parser=argparse.ArgumentParser(description='零碳校园负荷的冷启动预测与区间交付资料诊断')
    parser.add_argument('data')
    parser.add_argument('--as-of',required=True)
    args=parser.parse_args()
    print(output(describe(RecordBook.load(args.data),args.as_of)))

if __name__=='__main__':
    main()
