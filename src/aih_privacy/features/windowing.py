import numpy as np

def sliding_windows_indices(n_samples: int, win_size: int, step: int):
    for start in range(0, n_samples - win_size + 1, step):
        end = start + win_size
        yield start, end

def window_stats(x, win, step):
    x = np.asarray(x, float)
    maxs, means, stds = [], [], []
    for s, e in sliding_windows_indices(len(x), win, step):
        w = x[s:e]
        maxs.append(np.max(w))
        means.append(np.mean(w))
        stds.append(np.std(w))
    return np.array(maxs), np.array(means), np.array(stds)