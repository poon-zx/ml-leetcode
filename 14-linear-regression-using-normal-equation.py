import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	"""
	The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. 
	Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.
	"""
	X = np.array(X)
	# from a 1-D vector to a 2-D column, -1 means numpy will use however many rows are needed
	y = np.array(y).reshape(-1, 1)
	
	X_transpose = X.T
	theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
	
    # theta comes out of the normal equation as a (k x 1) numpy array
	# use flatten to convert it to one dimension (k, )
	# tolist to convert from np array to py list
	theta = np.round(theta, 4).flatten().tolist()

	return theta

def test_linear_regression_normal_equation() -> None:
    # Test case 1
    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    assert linear_regression_normal_equation(X, y) == [-0.0, 1.0], "Test case 1 failed"

    # Test case 2
    X = [[1, 3, 4], [1, 2, 5], [1, 3, 2]]
    y = [1, 2, 1]
    assert linear_regression_normal_equation(X, y) == [4.0, -1.0, -0.0], "Test case 2 failed"

if __name__ == "__main__":
    test_linear_regression_normal_equation()
    print("All linear_regression_normal_equation tests passed.")