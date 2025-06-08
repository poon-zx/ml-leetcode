import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
	d_k = Q.shape[1]

    attention_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    score_max = np.max(attention_scores, axis=1, keepdims=True)
    attention_weights = np.exp(attention_scores - score_max) / np.sum(np.exp(attention_scores - score_max), axis = 1, keepdims = True)

    return np.matmul(attention_weights, V)

def multi_head_attention(Q, K, V, n_heads):
	d_model = Q.shape[1]
    assert d_model % n_heads == 0 # ensure d_model is divisble by n_heads
    d_k = d_model // n_heads

    # Reshape Q, K, V to separate heads
    # original d_model is spilt into (n_heads, d_k)
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2) # transpose to (n_heads, seq_len, d_k) from (seq_len, n_heads, d_k)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)

    attentions = []
    for i in range(n_heads):
        attn = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i]) # Compute attention for the i-th head
        attentions.append(attn)

    # Concatenate along the columns axis
    attention_output = np.concatenate(attentions, axis=-1)
    return attention_output


