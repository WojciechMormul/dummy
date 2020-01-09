import tensorflow as tf
import numpy as np

DATASET_SIZE = 10
VOCAB_NUM = 20
CLASSES_NUM = 10
filename = 'test.tfrecord'

def _int64_feature(value):
	if type(value)!=list:
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(sequence, length, label):
  feature = {
      'sequence': _int64_feature(sequence),
      'length': _int64_feature(length),
      'label': _int64_feature(label)}

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

writer = tf.io.TFRecordWriter(filename)

for i in range(DATASET_SIZE):
	sequence = [np.random.randint(1, VOCAB_NUM) for _ in range(np.random.randint(2,10))]
	length = len(sequence)
	label = np.random.randint(0, CLASSES_NUM - 1)
	serialized_example = serialize_example(sequence, length, label)
	writer.write(serialized_example)

writer.close()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

feature_description = {
	'sequence': tf.io.VarLenFeature(tf.int64),
	#'sequence': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0) - doesn't work
	'length': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

def f(x):
	x['sequence'] = tf.sparse.to_dense(x['sequence'])
	return x

class Model(tf.keras.Model):
	
	def __init__(self):
		super(Model, self).__init__()
		self.embedding = tf.keras.layers.Embedding(VOCAB_NUM, 100)
		self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
		self.dense = tf.keras.layers.Dense(CLASSES_NUM)
		self.softmax = tf.keras.layers.Softmax()
		
	@tf.function
	def call(self, x):#, softmax = False):
		x = self.embedding(x)
		x = self.rnn(x)
		x = tf.nn.relu(x)
		x = self.dense(x)
		#if softmax == True:
		#	x = self.softmax(x)
		return x

dataset = tf.data.TFRecordDataset([filename])
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(10).batch(3)
dataset = dataset.map(f)

model = Model()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(model, sequence, label):
	with tf.GradientTape() as tape:
		predictions = model(sequence)
		loss = loss_object(label, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(label, predictions)

#for epoch in tf.range(tf.constant(100)):
for epoch in range(100):
	for batch in dataset:
		# length of a sequence doesn't matter now, because tf.nn.dynamic_rnn is depraced 
		# and tf.keras.RNN doesn't take length as input
		sequence = batch['sequence']
		label = batch['label']		
		#sequence = tf.zeros((3,8), dtype=tf.dtypes.int64)
		#label = tf.zeros((3), dtype=tf.dtypes.int64)
		train_step(model, sequence, label)
	print (train_loss.result())
	print (train_accuracy.result())
	train_loss.reset_states()
	train_accuracy.reset_states()
