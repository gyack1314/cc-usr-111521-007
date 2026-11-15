from .records import instant,number
from .model import target_times,school_info

def issued_weather(book,origin,school,target):
    cutoff=instant(origin);options=[]
    for row in book.rows('weather',origin):
        d=row['data']
        if d['school']==school and instant(d['target_time'])==instant(target) and instant(d['issued_at'])<=cutoff:
            options.append(row)
    if not options:return None
    return max(options,key=lambda r:(instant(r['data']['issued_at']),r['id']))

def known_history(book,origin,school):
    result=[]
    for row in book.rows('load',origin):
        d=row['data']
        if d['school']==school and instant(d['target_time'])<=instant(origin):
            value=number(d['kwh'])
            if value<0:raise ValueError('用电量不能为负')
            result.append({'source':row['id'],'target_time':d['target_time'],'kwh':value})
    return sorted(result,key=lambda r:instant(r['target_time']))

def feature_view(book,origin,school,horizons):
    info=school_info(book,origin,school);history=known_history(book,origin,school);rows=[]
    for horizon,target in target_times(origin,horizons):
        weather=issued_weather(book,origin,school,target)
        rows.append({'horizon':horizon,'target_time':target,'weather':weather,'school':info,'history_count':len(history)})
    return {'origin':origin,'school':school,'history':history,'future':rows,
            'fingerprint':book.fingerprint(['load','weather','school','calendar'],origin)}

def describe(book,as_of):
    return feature_view(book,'2026-11-01T00:00:00+00:00','north',[1,2])
