import numpy as np

def self_attention(Q, K, V):
    """
    Compute self-attention using the Q, K, V matrices.
    
    Args:
    Q, K, V: numpy arrays of shape (seq_len, d_model)
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_model)

    Given an input sequence X, self-attention computes three key components:
    Query (Q): Encodes how each token wants to attend to other tokens
    Key (K): Encodes what each token offers for attention
    Value (V): Encodes the information each token provides when attended to
    """
    # d_model is the size of the vector used to represent each token
    # e.g. each word could be represented by a 512-dimensional vector

    d_k = Q.shape[1]
    
    # calculate attention scores
    # scores.shape = (seq_len, seq_len)
    # each entry scores[i][j] represents the dot-product similarity between
    # query vector of token i and key vector of token j
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # apply softmax row-wise to get attention weights
    # The matrix tells you who is attending to whom, and how strongly
    # np.exp exponentiates each score to make all values positive and amplify differences
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    attention_output = np.matmul(attention_weights,V)

    return attention_output

def compute_qkv(X, W_q, W_k, W_v):
    """
    Compute Q, K, V matrices from input X and weights.
    
    Args:
    X: numpy array of shape (seq_len, d_model), input sequence
    W_q, W_k, W_v: numpy arrays of shape (d_model, d_model), learnt weight matrices for Q, K, V
    
    Returns:
    Q, K, V: numpy arrays of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def test_self_attention():
    # Test case 1: Basic functionality with computed Q, K, V
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    expected_output = np.array([[1.660477, 2.660477], [2.339523, 3.339523]])
    actual_output = self_attention(Q, K, V)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 1 failed")

    # Test case 2: Different input X and weight matrices
    X = np.array([[1, 1], [1, 0]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    expected_output = np.array([[3.00928465, 4.6790462 ],
                                [2.5 , 4.]])
    actual_output = self_attention(Q, K, V)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 2 failed")

    # Test case 3: Larger input
    X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    W_q = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    W_k = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    W_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    expected_output = np.array([[ 8.,10.,12.  ],
                            [ 8.61987385, 10.61987385, 12.61987385],
                            [ 7.38012615 , 9.38012615, 11.38012615]])
    actual_output = self_attention(Q, K, V)
    np.testing.assert_array_almost_equal(actual_output, expected_output, decimal=6, 
                                         err_msg="Test case 3 failed")

if __name__ == "__main__":
    test_self_attention()
    print("All self-attention tests passed.")