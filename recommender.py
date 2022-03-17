from surprise import Dataset, Reader, SVD
import numpy as np

import ray
from ray import serve
from fastapi import FastAPI

processed_path = 'data/processed.txt'

ray.init(address="auto", namespace="serve")
serve.start(detached=True, http_options={"host": "0.0.0.0"})  # the "0.0.0.0" allow external request
app = FastAPI()


@serve.deployment(name='recommend', num_replicas=4)
@serve.ingress(app)  # use the default name SVDRecommender
class SVDRecommender:

    def __init__(self, factors=100):
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

    @app.get("/{user_id}")
    def recommend(self, user_id):
        add_noise = True
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
        return ','.join(list(map(self.trainset.to_raw_iid, indices)))


if __name__ == '__main__':
    SVDRecommender.deploy()









