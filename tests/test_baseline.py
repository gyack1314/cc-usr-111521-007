import copy
import unittest
from pathlib import Path
from decimal import Decimal
from campus_forecast.records import RecordBook,number,instant

ASOF='2026-11-15T00:00:00Z'
SAMPLE=Path(__file__).resolve().parents[1]/'data'/'sample.json'

class Base(unittest.TestCase):
    def setUp(self):
        self.book=RecordBook.load(SAMPLE)

class RecordTests(Base):
    def test_revision_and_deletion(self):
        first={'kind':'x','id':'a','revision':1,'known_at':'2026-10-01T00:00:00Z','data':{'v':1}}
        second={**first,'revision':2,'known_at':'2026-11-01T00:00:00Z','data':{'v':2}}
        third={**first,'revision':3,'known_at':'2026-11-10T00:00:00Z','deleted':True}
        b=RecordBook({'schema':1,'records':[third,first,second]})
        self.assertEqual(b.rows('x','2026-10-20T00:00:00Z')[0]['data']['v'],1)
        self.assertEqual(b.rows('x','2026-11-05T00:00:00Z')[0]['data']['v'],2)
        self.assertEqual(b.rows('x',ASOF),[])
    def test_views_are_copies(self):
        kind=self.book.document['records'][0]['kind'];a=self.book.rows(kind,ASOF)
        a[0]['data']['mutated']=True
        self.assertNotIn('mutated',self.book.rows(kind,ASOF)[0]['data'])
    def test_duplicate_revision(self):
        d=self.book.document;d['records'].append(copy.deepcopy(d['records'][0]))
        self.assertRaises(ValueError,RecordBook,d)
    def test_strict_numbers(self):
        self.assertEqual(number('0.1')+number('0.2'),Decimal('0.3'))
        self.assertRaises(ValueError,number,'NaN')
        self.assertRaises(ValueError,instant,'2026-11-01T00:00:00')
    def test_canonical_fingerprint(self):
        d=self.book.document;d['records'].reverse();other=RecordBook(d)
        kinds={r['kind'] for r in d['records']}
        self.assertEqual(self.book.fingerprint(kinds,ASOF),other.fingerprint(kinds,ASOF))

from campus_forecast.operations import feature_view,issued_weather,known_history
from campus_forecast.model import target_times

class DomainTests(Base):
    def test_issue_cutoff(self):
        r=issued_weather(self.book,'2026-11-01T00:00:00Z','north','2026-11-01T01:00:00Z')
        self.assertEqual(r['id'],'first-1')
    def test_cold_school(self):
        r=feature_view(self.book,'2026-11-01T00:00:00Z','new',[1])
        self.assertEqual(r['history'],[]);self.assertIsNone(r['future'][0]['weather'])
    def test_horizons(self):
        self.assertRaises(ValueError,target_times,ASOF,[0])
        self.assertRaises(ValueError,target_times,ASOF,[1,1])
    def test_history_is_visible(self):
        self.assertEqual(known_history(self.book,'2026-11-01T00:00:00Z','north')[0]['kwh'],Decimal(20))
