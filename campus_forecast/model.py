from datetime import timedelta
from .records import instant,number

def target_times(origin,horizons):
    stamp=instant(origin)
    if len(set(horizons))!=len(horizons) or any(type(h) is not int or h<=0 for h in horizons):
        raise ValueError('预测距离须为不同的正整数小时')
    return [(h,(stamp+timedelta(hours=h)).isoformat()) for h in sorted(horizons)]

def school_info(book,as_of,school):
    found=[r for r in book.rows('school',as_of) if r['id']==school]
    if len(found)!=1 or number(found[0]['data']['floor_area'])<=0:raise ValueError('学校资料缺失')
    return found[0]['data']
