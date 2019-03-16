import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
FEATURE_TO_PREDICT = "Adj Close"

# Reading a CSV file
df = pd.read_csv("data/TITAN.NS.CSV")

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
	
	random.shuffle(sequential_data)


# Main DataFrame will hold required features
main_df = df[[f"{FEATURE_TO_PREDICT}"]]

main_df['future'] = main_df[f"{FEATURE_TO_PREDICT}"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{FEATURE_TO_PREDICT}"], main_df["future"]))

 # Let's create out-of-sample data

validation_set = main_df[-int(0.05 * len(main_df)):]

preprocess_df(main_df)
# train_x, train_y = preprocess_df(main_df)
# validation_x, validation_y = preprocess(main_df)

print()