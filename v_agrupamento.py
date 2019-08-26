from sklearn.cluster import KMeans

def agrupar_Kmeans(db,k):
	return KMeans(n_clusters=k).fit(db)

def get_groups(db,group_labels):
	grupos = []
	grupos = [db.groupby(group_labels.labels_).get_group(x) for x in set(group_labels.labels_)]
	return grupos