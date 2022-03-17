import pandas as pd
from collections import defaultdict

streaming_path = 'data/streamingData_1M.csv'
processed_path = 'data/processed.txt'


def normalize(x, u, l, nu, nl):
    x = max(l, x)
    x = min(u, x)
    x = (x - l) / (u - l)
    x = nl + (nu - nl) * x
    return x


def preprocess4surprise():
    """Preprocess streaming data for surprise reader"""
    df = pd.read_csv(streaming_path, sep=',')
    L = df.shape[0]
    df = df.iloc[:int(.8 * L)]
    pair2min = defaultdict(int)
    pair2rate = dict()

    for uid, mid, mi, rating in zip(df['user_id'].values, df['movie_id'].values, df['minute'].values, df['rating'].values):
        if not pd.isnull(mi):
            pair2min[(uid, mid)] += 1
        if not pd.isnull(rating):
            pair2rate[(uid, mid)] = rating
    for pair, mi in pair2min.items():
        if pair not in pair2rate:
            pair2rate[pair] = normalize(mi, 0, 110, 5, 1)
    out = []
    for (uid, mid), rate in pair2rate.items():
        out.append(';'.join((str(uid), mid, str(rate))))
    with open(processed_path, 'w') as f:
        f.write('\n'.join(out))



if __name__ == '__main__':
    preprocess4surprise()