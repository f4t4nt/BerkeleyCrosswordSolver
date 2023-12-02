import pickle
import numpy as np

with open('dpr_cache.pkl', 'rb') as f:
    dpr_cache = pickle.load(f)

def softmax_scores(scores):
    max_score = max(scores)
    score_sum = np.sum(np.exp(np.array(scores) - max_score))
    log_score_sum = np.log(score_sum) + max_score
    for i, score in enumerate(scores):
        scores[i] = np.exp(score - log_score_sum)
    return scores

def clean_dict(d):
    scores = list(softmax_scores(d[1]))
    return (d[0], scores)

def clean_cache(dpr_cache):
    for k in dpr_cache:
        # dpr_cache[k] = clean_dict(dpr_cache[k])
        tmp_map = {}
        for i, d in enumerate(dpr_cache[k][0]):
            tmp_map[d] = dpr_cache[k][1][i]
        dpr_cache[k] = tmp_map
    return dpr_cache

dpr_cache = clean_cache(dpr_cache)

with open('dpr_cache.pkl', 'wb') as f:
    pickle.dump(dpr_cache, f)