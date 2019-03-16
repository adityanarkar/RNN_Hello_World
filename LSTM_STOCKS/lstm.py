import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np


SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
FEATURE_TO_PREDICT = "Adj Close"


# Reading a CSV file


def classify(current, future):
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	df = df.drop('future', 1) # Dropping the predictions

	for col in df.columns:
		if col != "target":
			df[col] = df[col].pct_change()
			df.dropna(inplace=True)
			df[col] = preprocessing.scale(df[col].values)

	df.dropna(inplace=True)

	sequential_data = []
	prev_days = deque(maxlen=SEQ_LEN)

	for i in df.values: # df.values will not contain an index. It will return list of lists.
		prev_days.append([n for n in i[:-1]]) # n in i[:-1] will return a list w/o target
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]])
	
	# random.shuffle(sequential_data)

	buys = []
	sells = []

	for seq, target in sequential_data:
		if target == 1:
			buys.append([seq, target])
		elif target == 0:
			sells.append([seq, target])

	# Balancing the data

	lower = min(len(buys), len(sells))

	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys+sells # Balance data

	# random.shuffle(sequential_data)

	X = []
	y = []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), y


def prepareData():

	df = pd.read_csv("data/TITAN.NS.CSV")

	# Main DataFrame will hold required features
	main_df = df[[f"{FEATURE_TO_PREDICT}"]]

	main_df['future'] = main_df[f"{FEATURE_TO_PREDICT}"].shift(-FUTURE_PERIOD_PREDICT)

	main_df['target'] = list(map(classify, main_df[f"{FEATURE_TO_PREDICT}"], main_df["future"]))

	# Let's create training and testing samples
	train, validation_set = main_df[:-int(0.1 * len(main_df))], main_df[-int(0.1 * len(main_df)):]

	# preprocess_df(main_df)
	train_x, train_y = preprocess_df(train)
	validation_x, validation_y = preprocess_df(validation_set)

	return train_x, train_y, validation_x, validation_y


