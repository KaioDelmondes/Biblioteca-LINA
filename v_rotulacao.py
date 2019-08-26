from collections import Counter
from man_dados import named

class rotulador(object):
	def __init__ (self, cluster, holdout_val, V):
		self.holdout_val = holdout_val
		self.V = V

		self.medias = 100*(self.holdout_val.mean(1))
		self.valor_max = self.medias.max()

		self.limite = ((100-self.V)*self.valor_max)/100

		self.titulos = named(cluster)
		self.cluster = cluster.as_matrix()

	def rotular_bd_naoDiscretizada(self):
		#Calcula a média de acerto para cada atributo do cluster  
		self.medias.sort()
		self.medias = self.medias[::-1]
		
		rotulo = ''

		#Para cada atributo
		for i in range(0, self.medias.shape[0]): 
			# testa se a média do atributo está fora do parametro V
			if self.medias[i] >= self.limite:
				# mr : tupla com o numero que mais aparece no atributo avaliado e quantas ocorrencias			
				mr = Counter(self.cluster[:,i]).most_common(1) 
				# Rotulo: Atrbituto i: valor_mais_frequente  | Relevancia : relevancia%
				rotulo +=  self.titulos[i] +" :" + str (mr[0][0]) + " | Relevância: " + str(self.medias[i]) + "%\n"
		return rotulo

	def rotular_bd_discretizada(self, infor):
		#Calcula a média de acerto para cada atributo do cluster
		rotulo = ''
		for i in range(0, self.medias.shape[0]):
			if self.medias[i] >= self.limite:
				mr = Counter(self.cluster[:,i]).most_common(1) #retorna uma tupla com o numero que mais aparece no atributo avaliado e quantas ocorrencias
				mr = int(mr[0][0]) 
				rotulo += self.titulos[i] + ": " + str(infor[i][mr]) + " ~ " + str(infor[i][mr + 1]) + " | Relevância: " + str(self.medias[i]) + "%\n"
		return rotulo


def rotula_grupo( cluster, holdout_val, V, infor):
rotulador_ = rotulador(cluster, holdout_val, V)
	if infor:
		rotulo = rotulador_.rotular_bd_discretizada(infor)
	else:
		rotulo = rotulador_.rotular_bd_naoDiscretizada()
	return rotulo

def rotular( grupos, classificacao_infor, V, discretizacao_infor):
	rotulo = []
	for i in range(0, len(grupos)):	
		rotulo.append(rotula_grupo(grupos[i], classificacao_infor[i], V, discretizacao_infor))
	return rotulo