"""保留修订历史的只读资料视图。"""
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import hashlib
import json

def number(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('数值须有限')
    return result

def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('资料时点须包含时区')
    return result

class RecordBook:
    def __init__(self, document):
        self.document = deepcopy(document)
        if document.get('schema') != 1:
            raise ValueError('不支持的资料版本')
        self._rows = document['records']
        seen = set()
        for row in self._rows:
            key = (row['kind'], row['id'], row['revision'])
            if key in seen:
                raise ValueError('重复修订身份')
            seen.add(key)
            if not isinstance(row['revision'], int) or row['revision'] < 1:
                raise ValueError('修订号须为正整数')
            instant(row['known_at'])
            if not isinstance(row.get('data', {}), dict):
                raise ValueError('记录正文须为对象')

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            return cls(json.load(stream))

    def rows(self, kind, as_of):
        cutoff = instant(as_of)
        latest = {}
        for row in self._rows:
            if row['kind'] != kind or instant(row['known_at']) > cutoff:
                continue
            previous = latest.get(row['id'])
            rank = (instant(row['known_at']), row['revision'])
            if previous is None or rank > (instant(previous['known_at']), previous['revision']):
                latest[row['id']] = row
        return [deepcopy(row) for key, row in sorted(latest.items()) if not row.get('deleted', False)]

    def fingerprint(self, kinds, as_of):
        payload = {kind: self.rows(kind, as_of) for kind in sorted(set(kinds))}
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(body.encode()).hexdigest()

def output(value):
    def convert(item):
        if isinstance(item, Decimal):
            return str(item)
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False)
