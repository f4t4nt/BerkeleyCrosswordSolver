import pickle
import numpy as np
# import matplotlib.pyplot as plt

with open("./285/pkl/all_runs.pkl", "rb") as f:
    all_runs = pickle.load(f)

all_runs_T = [[] for _ in range(10)]

for run in all_runs:
    for i in range(min(len(run), len(all_runs_T))):
        all_runs_T[i].append(run[i])

for i in range(len(all_runs_T)):
    all_runs_T[i].sort()

def pick_scores_at_i(i, percentiles, noise):
    scores = []
    for p in percentiles:
        idx = int(p * len(all_runs_T[i]))
        idx += np.random.normal(0, noise)
        idx = int(idx)
        idx = max(0, idx)
        idx = min(idx, len(all_runs_T[i]) - 1)
        scores.append(all_runs_T[i][idx])
    return scores

def pick_scores(cnt=1):
    lo, mid, hi = 0, 0.01, 0.02
    scores = [[] for _ in range(cnt)]
    for i in range(len(all_runs_T)):
        p = i / len(all_runs_T)
        # new_scores = pick_scores_at_i(i, [0.5 * (1 - p) ** 6 + mid * (1 - p) + lo * p, 0.5 * (1 - p) ** 6 + mid, 0.5 * (1 - p) ** 6 + mid * (1 - p) + hi * p], 10)
        new_scores = pick_scores_at_i(i, [0.2], 0)
        for j in range(cnt):
            scores[j].append(new_scores[j])
    return scores

scores = pick_scores()
print(scores)