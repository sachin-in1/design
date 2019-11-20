"""
This is a implementation of Word2Vec using numpy. Uncomment the print functions to see Word2Vec in action! Also remember to change the number of epochs and set training_data to training_data[0] to avoid flooding your terminal. A Google Sheet implementation of Word2Vec is also available here - https://docs.google.com/spreadsheets/d/1mgf82Ue7MmQixMm2ZqnT1oWUucj6pEcd2wDs_JgHmco/edit?usp=sharing
Have fun learning!
Author: Derek Chia
Email: derek@derekchia.com
"""

import numpy as np
from collections import defaultdict

## Randomly initialise
getW1 = [[-0.0017610127, -0.0008330198, 0.0037335064, 0.0044187154, -0.000893576, 0.0028819032, -0.004903138, 0.002563034, -0.003466659, 0.0008959454, -0.0017700191, 0.003531669, 0.0018644591, -0.0009750846, -0.0035052267, -0.0049308236, -0.002639563, -0.001888581, 0.00076190993, -0.0013881406, -0.0016899507, 0.0031733417, 0.0008759684, 0.0040001445, -0.0014631036, 0.0043554534, 0.00087489514, -0.0023794456, 0.0023045682, 0.0018353256, -0.0045549353, -0.002546716, 0.004217902, 0.0041161915, -5.376815e-05, -0.002270047, 0.0016562288, -0.004541938, 0.0025901091, -0.0004936552, -0.0045318445, 0.0014925174, 0.0032492226, 0.0022884698, -0.0007430835, -0.00089463976, -0.00082940876, 0.00086756394, 0.0040442646, 0.0029277175, 0.00066780613, 0.001493269, -0.0012386364, -0.0005698611, 0.00050490635, 0.001244483, -0.0015087216, -0.0012871558, 0.0047525666, -0.0021011704, 0.00053441944, 0.0021138354, -0.0029833957, 0.00028362102, -0.0023581518, -0.0030776896, -0.0028096628, 0.0040602856, -0.0032284183, 0.004590996, -0.0020473506, 0.0035585247, -0.0019253556, -0.00043411844, -0.0035071431, -0.0010036575, 0.003429804, 0.0048693516, 0.00014534322, -0.00211264, -0.002266037, -0.0026173738, 0.00026544093, -0.0028365224, -0.0019675978, 0.00094564824, 0.0046581337, -0.0044808206, -0.004625488, -0.004635902, 0.00057746726, -0.0035385292, -0.0030641926, 0.0030930345, -0.0042488016, -0.003814588, 0.0049046758, -0.0040246053, -0.0010995828, 0.0031983347],
		[0.00036156387, 0.0010682167, -0.0010308616, -0.0019382038, -0.004618606, 0.0013638993, 0.00078894384, -0.0014671609, 0.004425118, 0.004417468, 0.00045506892, -0.0020692598, 0.004065457, -0.0036094463, -0.00302995, -0.0044123214, 0.0033492076, 0.0013892442, -0.0042852936, 0.0037480427, 0.004346591, -0.0011719287, -0.0013133038, 0.0013068955, -0.0031381953, 0.004906961, 0.0028233298, 0.0031934616, -0.0022448837, 0.00011932392, 0.0048393463, -0.0046925037, -0.0036888046, -0.0026013998, -0.0014884258, -0.0019935167, -0.0026723559, -0.0031095457, 0.0032059292, 0.0036727504, -0.00290165, 0.001846062, -0.0012719407, -0.002454337, 0.0016873394, -0.0026243222, -0.0037982988, -0.0015448632, 0.0021020113, 0.0012299946, 0.0009617684, -0.0002006072, 0.0022099218, -0.0039009964, 0.002569506, 0.003118239, 0.0039542033, 0.0043327864, 0.0033249548, 0.00397863, -0.000103656996, -0.0044851196, -0.0026745172, 0.002267192, -0.0012654141, -0.0034608005, -0.0026225678, -0.00077755784, -0.0004944914, 0.0040423595, 9.930666e-05, -0.00035855084, 0.0046567535, 0.0035449853, 0.0024499856, 0.0026082445, 0.0027717832, -0.0010435606, 0.0033203743, -0.0043739392, 0.0011623811, 0.00493896, 0.005049161, 0.0018619401, 0.0036583564, -0.0032303121, -0.0041065593, 0.0010996294, -0.0022788346, 0.0038857716, -0.0011292284, -0.0004869222, -0.0046112356, 0.00025858893, 0.001163639, 0.00050099933, -0.0016287448, 0.0009662275, 0.0017630983, -0.0020982805],
		[-0.576, 0.658, -0.582, -0.112, 0.662, 0.051, -0.401, -0.921, -0.158, 0.529],
		[0.517, 0.436, 0.092, -0.835, -0.444, -0.905, 0.879, 0.303, 0.332, -0.275],
		[0.859, -0.890, 0.651, 0.185, -0.511, -0.456, 0.377, -0.274, 0.182, -0.237],
		[0.368, -0.867, -0.301, -0.222, 0.630, 0.808, 0.088, -0.902, -0.450, -0.408],
		[0.728, 0.277, 0.439, 0.138, -0.943, -0.409, 0.687, -0.215, -0.807, 0.612],
		[0.593, -0.699, 0.020, 0.142, -0.638, -0.633, 0.344, 0.868, 0.913, 0.429],
		[0.447, -0.810, -0.061, -0.495, 0.794, -0.064, -0.817, -0.408, -0.286, 0.149]]

getW2 = [[-0.868, -0.406, -0.288, -0.016, -0.560, 0.179, 0.099, 0.438, -0.551],
		[-0.395, 0.890, 0.685, -0.329, 0.218, -0.852, -0.919, 0.665, 0.968],
		[-0.128, 0.685, -0.828, 0.709, -0.420, 0.057, -0.212, 0.728, -0.690],
		[0.881, 0.238, 0.018, 0.622, 0.936, -0.442, 0.936, 0.586, -0.020],
		[-0.478, 0.240, 0.820, -0.731, 0.260, -0.989, -0.626, 0.796, -0.599],
		[0.679, 0.721, -0.111, 0.083, -0.738, 0.227, 0.560, 0.929, 0.017],
		[-0.690, 0.907, 0.464, -0.022, -0.005, -0.004, -0.425, 0.299, 0.757],
		[-0.054, 0.397, -0.017, -0.563, -0.551, 0.465, -0.596, -0.413, -0.395],
		[-0.838, 0.053, -0.160, -0.164, -0.671, 0.140, -0.149, 0.708, 0.425],
		[0.096, -0.995, -0.313, 0.881, -0.402, -0.631, -0.660, 0.184, 0.487]]

class word2vec():

	def __init__(self):
		self.n = settings['n']
		self.lr = settings['learning_rate']
		self.epochs = settings['epochs']
		self.window = settings['window_size']

	def generate_training_data(self, settings, corpus):
		# Find unique word counts using dictonary
		word_counts = defaultdict(int)
		for row in corpus:
			for word in row:
				word_counts[word] += 1
		#########################################################################################################################################################
		# print(word_counts)																																	#
		# # defaultdict(<class 'int'>, {'natural': 1, 'language': 1, 'processing': 1, 'and': 2, 'machine': 1, 'learning': 1, 'is': 1, 'fun': 1, 'exciting': 1})	#
		#########################################################################################################################################################

		## How many unique words in vocab? 9
		self.v_count = len(word_counts.keys())
		#########################
		# print(self.v_count)	#
		# 9						#
		#########################

		# Generate Lookup Dictionaries (vocab)
		self.words_list = list(word_counts.keys())
		#################################################################################################
		# print(self.words_list)																		#
		# ['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'exciting']	#
		#################################################################################################
		
		# Generate word:index
		self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
		#############################################################################################################################
		# print(self.word_index)																									#
		# # {'natural': 0, 'language': 1, 'processing': 2, 'and': 3, 'machine': 4, 'learning': 5, 'is': 6, 'fun': 7, 'exciting': 8}	#
		#############################################################################################################################

		# Generate index:word
		self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
		#############################################################################################################################
		# print(self.index_word)																									#
		# {0: 'natural', 1: 'language', 2: 'processing', 3: 'and', 4: 'machine', 5: 'learning', 6: 'is', 7: 'fun', 8: 'exciting'}	#
		#############################################################################################################################

		training_data = []

		# Cycle through each sentence in corpus
		for sentence in corpus:
			sent_len = len(sentence)

			# Cycle through each word in sentence
			for i, word in enumerate(sentence):
				# Convert target word to one-hot
				w_target = self.word2onehot(sentence[i])

				# Cycle through context window
				w_context = []

				# Note: window_size 2 will have range of 5 values
				for j in range(i - self.window, i + self.window+1):
					# Criteria for context word 
					# 1. Target word cannot be context word (j != i)
					# 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
					# 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
					if j != i and j <= sent_len-1 and j >= 0:
						# Append the one-hot representation of word to w_context
						w_context.append(self.word2onehot(sentence[j]))
						# print(sentence[i], sentence[j]) 
						#########################
						# Example:				#
						# natural language		#
						# natural processing	#
						# language natural		#
						# language processing	#
						# language append 		#
						#########################
						
				# training_data contains a one-hot representation of the target word and context words
				#################################################################################################
				# Example:																						#
				# [Target] natural, [Context] language, [Context] processing									#
				# print(training_data)																			#
				# [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]]]	#
				#################################################################################################
				training_data.append([w_target, w_context])

		return np.array(training_data)

	def word2onehot(self, word):
		# word_vec - initialise a blank vector
		word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
		#############################
		# print(word_vec)			#
		# [0, 0, 0, 0, 0, 0, 0, 0]	#
		#############################

		# Get ID of word from word_index
		word_index = self.word_index[word]

		# Change value from 0 to 1 according to ID of the word
		word_vec[word_index] = 1

		return word_vec

	def train(self, training_data):
		# Initialising weight matrices
		# np.random.uniform(HIGH, LOW, OUTPUT_SHAPE)
		# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.uniform.html
		self.w1 = np.array(getW1)
		self.w2 = np.array(getW2)
		# self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
		# self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
		
		# Cycle through each epoch
		for i in range(self.epochs):
			# Intialise loss to 0
			self.loss = 0
			# Cycle through each training sample
			# w_t = vector for target word, w_c = vectors for context words
			for w_t, w_c in training_data:
				# Forward pass
				# 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
				y_pred, h, u = self.forward_pass(w_t)
				#########################################
				# print("Vector for target word:", w_t)	#
				# print("W1-before backprop", self.w1)	#
				# print("W2-before backprop", self.w2)	#
				#########################################

				# Calculate error
				# 1. For a target word, calculate difference between y_pred and each of the context words
				# 2. Sum up the differences using np.sum to give us the error for this particular target word
				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				#########################
				# print("Error", EI)	#
				#########################

				# Backpropagation
				# We use SGD to backpropagate errors - calculate loss on the output layer 
				self.backprop(EI, h, w_t)
				#########################################
				#print("W1-after backprop", self.w1)	#
				#print("W2-after backprop", self.w2)	#
				#########################################

				# Calculate loss
				# There are 2 parts to the loss function
				# Part 1: -ve sum of all the output +
				# Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
				# Note: word.index(1) returns the index in the context word vector with value 1
				# Note: u[word.index(1)] returns the value of the output layer before softmax
				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
				
				#############################################################
				# Break if you want to see weights after first target word 	#
				# break 													#
				#############################################################
			print('Epoch:', i, "Loss:", self.loss)

	def forward_pass(self, x):
		# x is one-hot vector for target word, shape - 9x1
		# Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
		h = np.dot(x, self.w1)
		# Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
		u = np.dot(h, self.w2)
		# Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
		y_c = self.softmax(u)
		return y_c, h, u

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)

	def backprop(self, e, h, x):
		# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
		# Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
		# Going backwards, we need to take derivative of E with respect of w2
		# h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
		# x - shape 9x1, w2 - 10x9, e.T - 9x1
		dl_dw2 = np.outer(h, e)
		dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
		########################################
		# print('Delta for w2', dl_dw2)			#
		# print('Hidden layer', h)				#
		# print('np.dot', np.dot(self.w2, e.T))	#
		# print('Delta for w1', dl_dw1)			#
		#########################################

		# Update weights
		self.w1 = self.w1 - (self.lr * dl_dw1)
		self.w2 = self.w2 - (self.lr * dl_dw2)

	# Get vector from word
	def word_vec(self, word):
		w_index = self.word_index[word]
		v_w = self.w1[w_index]
		return v_w

	# Input vector, returns nearest word(s)
	def vec_sim(self, word, top_n):
		v_w1 = self.word_vec(word)
		word_sim = {}

		for i in range(self.v_count):
			# Find the similary score for each word in vocab
			v_w2 = self.w1[i]
			theta_sum = np.dot(v_w1, v_w2)
			theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
			theta = theta_sum / theta_den

			word = self.index_word[i]
			word_sim[word] = theta

		words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

		for word, sim in words_sorted[:top_n]:
			print(word, sim)

#####################################################################
settings = {
	'window_size': 2,			# context window +- center word
	'n': 10,					# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 150,				# number of training epochs
	'learning_rate': 0.01		# learning rate
}

text = "Rapid Automatic Keyword Extraction (RAKE) is an algorithm to automatically extract keywords from documents. Keywords are sequences of one or more words that, together, provide a compact representation of content. RAKE is a well-known and widely used NLP technique, but its concrete application depends a lot on factors like the language in which the content is written, the domain of the content and the purpose of the keywords."

# Note the .lower() as upper and lowercase does not matter in our implementation
# [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
corpus = [[word.lower() for word in text.split()]]

# Initialise object
w2v = word2vec()

# Numpy ndarray with one-hot representation for [target_word, context_words]
training_data = w2v.generate_training_data(settings, corpus)

# Training
w2v.train(training_data)

# Get vector for word
word = "natural"
vec = w2v.word_vec(word)
print(word, vec)

# Find similar words
w2v.vec_sim("natural", 10)