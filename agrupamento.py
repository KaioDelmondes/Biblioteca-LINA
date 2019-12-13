import interfaces.scikitlearn_interface as scikitlearn_interface
from pandas import DataFrame

def Kmeans(db,quantidade_de_grupos):
	data_df = DataFrame(db)
	kmeans_ret = scikitlearn_interface.KMeans_sklearn(data_df,quantidade_de_grupos)
	col_cluster = kmeans_ret.labels_
	db_agrupado = data_df.assign(col_cluster=col_cluster)

	return db_agrupado