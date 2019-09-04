import pandas

def read_csv(path, separador=","):
	bd = pandas.read_csv(path,sep=',',parse_dates=True)
	return bd

def make_df(data):
	return pandas.DataFrame(data)

def get_groups(DataFrameGroupedBy, k=None):
	if k is None:
		k = len(DataFrameGroupedBy)
	grupos = [DataFrameGroupedBy.get_group(i) for i in range(k)]
	return grupos

