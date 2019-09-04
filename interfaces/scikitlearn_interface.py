from sklearn.cluster import KMeans

def KMeans_sklearn(db,k):
	try:
		kmeans_obj = KMeans(n_clusters=k).fit(db)	
		return kmeans_obj
	except Exception as e:
		raise e