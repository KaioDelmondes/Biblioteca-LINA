from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

def KMeans_sklearn(db,k):
	try:
		kmeans_obj = KMeans(n_clusters=k).fit(db)	
		return kmeans_obj
	except Exception as e:
		raise e

class MLP_sklearn:
	def __init__(self, camadas_ocultas):
		self.estimator = MLPClassifier(hidden_layer_sizes=camadas_ocultas)

	def treinar(self, entradas, attbr_classe):
		self.estimator.fit(entradas, attbr_classe.astype(int))
	
	def estimar_um(self, carac_entrada):
		return self.estimator.predict(carac_entrada.reshape(1,-1))

	def estimar_n(self, carac_entrada):
		return self.estimator.predict(carac_entrada)
	
	def precisao(self, carac_entrada, saida_real):
		prec_perc = self.estimator.score(carac_entrada, saida_real.astype(int))*100
		return prec_perc