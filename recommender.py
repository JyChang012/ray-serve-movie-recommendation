import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, SVD
import numpy as np
from os.path import exists
from sklearn.metrics import label_ranking_average_precision_score
from timeit import default_timer as timer

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


class SVDRecommender:

    def __init__(self, factors=100):
        # preprocess4surprise()
        # TODO: maybe add some lock to stop multiprocessing here, or run some script beforehand?
        reader = Reader(line_format='user item rating', sep=';')
        assert not reader.with_timestamp
        data = Dataset.load_from_file(processed_path, reader=reader)
        trainset = data.build_full_trainset()
        algo = SVD(n_factors=factors)
        algo.fit(trainset)
        self.algo = algo
        self.trainset = trainset

        # use when user no found in train set
        self.mean_pu = np.mean(self.algo.pu, axis=0)
        self.mean_bu = np.mean(self.algo.bu)

        print(self.algo.pu.shape, self.algo.qi.shape)

    def recommend(self, user_id, add_noise=True):
        user_id = str(user_id)
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
            rst = self.algo.pu[inner_uid] @ self.algo.qi.T + self.algo.bi + self.algo.bu[inner_uid]
        except ValueError:
            # if user not in train set
            rst = self.mean_pu @ self.algo.qi.T + self.algo.bi + self.mean_bu
        if add_noise:
            rst += .05 * np.random.randn(self.algo.qi.shape[0])
        indices = np.argsort(rst)[:-21:-1]
        # assert len(indices) == 20, len(indices)
        return list(map(self.trainset.to_raw_iid, indices))


def evaluate():
    df = pd.read_csv(streaming_path, sep=',')
    L = df.shape[0]
    df = df.iloc[int(.8 * L):]
    user2movie = defaultdict(set)
    pair2min = defaultdict(int)
    start = timer()
    model = SVDRecommender(factors=500)
    end = timer()
    print(f'train time: {end - start}')
    # prediction = model.algo.pu @ model.algo.qi.T + model.algo.bi + model.algo.bu.reshape([-1, 1])
    # ground_truth = np.zeros_like(prediction, dtype=int)

    for _, r in df[['user_id', 'movie_id', 'minute']].iterrows():
        uid = r['user_id']
        mid = r['movie_id']
        mi = r['minute']
        if not pd.isnull(mi):
            pair2min[(uid, mid)] += 1
    for (uid, mid), min in pair2min.items():
        if min >= 0:
            user2movie[uid].add(mid)
            '''
            try:
                ground_truth[model.trainset.to_inner_uid(uid), model.trainset.to_inner_iid(mid)] = 1
            except ValueError:
                continue
            '''
    count = 0
    total = 0
    tt = 0
    for uid, movies in user2movie.items():
        start = timer()
        recommended = model.recommend(uid)
        end = timer()
        recommended = set(recommended)
        total += len(recommended.intersection(movies)) / len(movies)
        count += 1
        tt += (end - start)
    print(f'inference time: {tt / count}')
    # print(label_ranking_average_precision_score(ground_truth, prediction))
    print(total / count)


if __name__ == '__main__':
    preprocess4surprise()
    evaluate()
    pass








