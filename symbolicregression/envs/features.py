from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import special_ortho_group
from sklearn.preprocessing import StandardScaler

class CustomDistribution(ABC):
    def __init__(self):
        pass

    def sample(
        self, rng: np.random.RandomState, n: Tuple[int, int]
    ) -> np.ndarray:
        pass

    def min(self) -> float:
        pass

    def max(self) -> float:
        pass


class CorrelatedGaussian(CustomDistribution):
    def __init__(self, a, b, centroids=10):

        self.a, self.b = a, b
        self.centroids = centroids

    def sample(self, rng: np.random.RandomState, n: Tuple[int, int]) -> List[float]:
        a, b, = self.a, self.b
        n_centroids = rng.randint(1, self.centroids + 1)
        n_points, input_dimension = n
        means = rng.randn(n_centroids, input_dimension,)
        means = rng.uniform(0.0, a, size=(n_centroids, input_dimension))
        covariances = rng.uniform(0.0, b, size=(n_centroids, input_dimension))
        covariances += np.identity(1)
        rotations = [
            special_ortho_group.rvs(input_dimension)
            if input_dimension > 1
            else np.identity(1)
            for i in range(n_centroids)
        ]
        weights = rng.uniform(0, 1.0, size=(n_centroids,))
        weights /= np.sum(weights)
        n_points_comp = rng.multinomial(n_points, weights)
        input = np.vstack(
            [
                rng.multivariate_normal(mean, np.diag(covariance), int(sample))
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )
        return input

    def mean(self) -> float:
        return self.a

    def std(self) -> float:
        return self.b

class MixtureOfDistributions(CustomDistribution):
    def __init__(self, distributions: List[CustomDistribution], probs=List[float]):
        assert len(distributions) == len(probs)
        assert np.abs(sum(probs) - 1.0) < 1e-7
        self.distributions = distributions
        self.probs = probs

    def sample(self, rng: np.random.RandomState, n: Tuple[int, int]) -> List[float]:
        n_points, dim = n
        distribution_counts = rng.multinomial(n=n_points, pvals=self.probs)
        samples = []
        for distribution, count in zip(self.distributions, distribution_counts):
            samples.append(distribution.sample(rng, [count, dim]))
        return np.concatenate(samples,0)


class CorrelatedUniform(CustomDistribution):
    def __init__(self, a, b, centroids=10):
        self.a, self.b = a, b
        self.centroids = centroids

    def sample(self, rng: np.random.RandomState, n: Tuple[int, int]) -> List[float]:
        a, b, = self.a, self.b
        n_centroids = rng.randint(1, self.centroids + 1)
        n_points, input_dimension = n
        means = rng.uniform(0.0, a, size=(n_centroids, input_dimension))
        covariances = rng.uniform(0.0, b, size=(n_centroids, input_dimension))
        covariances += np.identity(1)
        rotations = [
            special_ortho_group.rvs(input_dimension)
            if input_dimension > 1
            else np.identity(1)
            for i in range(n_centroids)
        ]
        weights = rng.uniform(0, 1.0, size=(n_centroids,))
        weights /= np.sum(weights)
        n_points_comp = rng.multinomial(n_points, weights)
        input = np.vstack(
            [
                (
                    mean
                    + rng.uniform(-1, 1, size=(sample, input_dimension))
                    * np.sqrt(covariance)
                )
                @ rotation
                for (mean, covariance, rotation, sample) in zip(
                    means, covariances, rotations, n_points_comp
                )
            ]
        )
        return input

    def mean(self) -> float:
        return self.a

    def std(self) -> float:
        return self.b

def sample_features_from_mixture(rng, feature_dim: int, n: int = None, normalize: bool = True) -> np.ndarray:

    distribution = MixtureOfDistributions(
        [CorrelatedGaussian(3, 5), CorrelatedUniform(3, 5)], [0.5, 0.5]
    )
    x = distribution.sample(rng, n=(n, feature_dim))
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def sample_features_from_uniform(rng, limits: Tuple[float], feature_dim: int, n: int = None) -> np.ndarray:
    assert limits[0] < limits[1]
    x = rng.uniform(low=limits[0], high=limits[1], size=(n, feature_dim))
    return x