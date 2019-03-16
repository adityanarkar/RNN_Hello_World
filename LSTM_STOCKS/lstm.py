import pandas as pd


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

# Main DataFrame will hold required features
main_df = df[[f"{FEATURE_TO_PREDICT}"]]

main_df['future'] = main_df[f"{FEATURE_TO_PREDICT}"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{FEATURE_TO_PREDICT}"], main_df["future"]))

print(main_df.head(10))