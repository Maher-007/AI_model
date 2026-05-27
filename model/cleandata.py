import pandas as pd

def clean_data(data):
	"""Clean the provided pandas DataFrame and return it.

	Expects a DataFrame containing a 'diagnosis' column. Drops
	optional columns if present and maps labels to integers.
	"""
	data = data.copy()
	drop_cols = [c for c in ['id', 'Unnamed: 32'] if c in data.columns]
	if drop_cols:
		data = data.drop(drop_cols, axis=1)
	if 'diagnosis' in data.columns:
		data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
	return data