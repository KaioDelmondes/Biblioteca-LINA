from pandas import DataFrame, read_csv

def ler_csv(path, separador=","):
	bd = read_csv(path,sep=',',parse_dates=True)
	return bd

def make_df(data):
	return DataFrame(data)

def get_groups(DataFrameGroupedBy, k=None):
	if k is None:
		k = len(DataFrameGroupedBy)
	grupos = [DataFrameGroupedBy.get_group(i) for i in range(k)]
	return grupos

def is_matriz(array):
	try:
		array.shape[1]
		return True
	except IndexError:
		return False