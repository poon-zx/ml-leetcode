import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
	scores_np = np.array(scores)
	probabilities = np.exp(scores_np) / np.sum(np.exp(scores_np))
	return probabilities.tolist()

def test_softmax():
    # Test case 1
    scores = [1, 2, 3]
    expected_output = [0.0900, 0.2447, 0.6652]
    assert softmax(scores) == expected_output, "Test case 1 failed"

    # Test case 2
    scores = [1, 1, 1]
    expected_output = [0.3333, 0.3333, 0.3333]
    assert softmax(scores) == expected_output, "Test case 2 failed"

    # Test case 3
    scores = [-1, 0, 5]
    expected_output = [0.0025, 0.0067, 0.9909]
    assert softmax(scores) == expected_output, "Test case 3 failed"

if __name__ == "__main__":
    test_softmax()
    print("All softmax tests passed.")