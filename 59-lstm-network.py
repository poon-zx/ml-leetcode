import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		h, c = initial_hidden_state.copy(), initial_cell_state.copy()
        hidden_history = []

        for t in range(x.shape[0]):
            x_t = x[t].reshape(-1, 1) # (input_size, 1)
            concat = np.vstack((h, x_t)) # (hidden+input, 1)

            # Forget gate
            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

            # Input gate
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            
            # Candidate
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

            # Cell state update
            c = ft * c + it * c_tilde

            # Output gate
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            # Hidden state update
            h = ot * np.tanh(c)
            hidden_history.append(h)
        
        return np.array(hidden_history), h, c
