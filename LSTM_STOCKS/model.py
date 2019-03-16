import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
import lstm


EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{lstm.SEQ_LEN}-SEQ-{lstm.FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
train_x, train_y, validation_x, validation_y = lstm.prepareData()

print(train_x.shape[1:])

model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
	optimizer =opt,
	metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "models/LSTM-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint(filepath, monitor=f"val_acc", verbose=1, save_best_only=False, save_weights_only=False, mode='max', period=1)

history = model.fit(train_x, 
		train_y,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		validation_data=(validation_x, validation_y),
		callbacks=[tensorboard, checkpoint])

# model.save()