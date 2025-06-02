import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    features_np = np.array(features)
    weights_np = np.array(weights)
    weighted_sum = np.matmul(features_np, weights_np) + bias
    probabilities = 1 / (1 + np.exp(-1 * weighted_sum))
    mse = 0
    for i in range(len(probabilities)):
        mse += abs(probabilities[i] - labels[i]) ** 2
    mse /= len(labels)
    probabilities = [round(num, 4) for num in probabilities]
    return probabilities, round(mse, 4)