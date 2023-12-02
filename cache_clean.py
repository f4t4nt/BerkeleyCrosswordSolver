import re
import pickle

with open('dpr_cache.pkl', 'rb') as f:
    dpr_cache = pickle.load(f)

def clean_string(s, remove_spaces=True):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    return s

def clean_dict(d):
    d[0][0] = [clean_string(s) for s in d[0][0]]
    return (d[0][0], list(d[1][0]))

def clean_cache(dpr_cache):
    for k in dpr_cache:
        dpr_cache[k] = clean_dict(dpr_cache[k])
    return dpr_cache

dpr_cache = clean_cache(dpr_cache)

with open('dpr_cache.pkl', 'wb') as f:
    pickle.dump(dpr_cache, f)