import numpy as np
from copy import copy
from sklearn.neural_network import MLPClassifier as mlp #Modelo classificador com Redes Neurais (MLP)
from sklearn import tree #Modelo classificador com Árvore de decisão (CART)

class classificador(object):
	"""Construtor da classe 'classificador'. 
	Parâmetros:
		metodo: Método de Aprendizagem Supervisionada de preferência(MLP ou Árvore de Decisão).
		cluster: conjunto de dados a serem usados na rede neural.
		perc_trein: a porcentagem de dados que será usada no treinamento da rede. O restante será usado como conjunto de teste.
	"""
	def __init__(self, metodo, cluster, perc_trein):
		super(classificador, self).__init__()
		self.cluster = cluster
		self.porc_treino = int(perc_trein*(self.cluster.shape[0]/100)) #Usa o perc_trein para definir quantas tuplas serão usadas no treino do modelo classificador
		self.resultados = np.zeros((cluster.shape[1],10), dtype = np.float64) #Matriz de resultados retornados pelo método de classificação. Usada na linha 40.
		self.classificar(metodo)

	"""
		Método principal. Não recebe nada como parâmetro pois usa somente a matriz de dados(cluster) como entrada
		para os métodos 'fit' e 'score' pertencentes ao objeto 'clf'
	"""	
	def classificar(self, metodo):
		self.cluster = np.asarray(self.cluster, dtype = 'int32')
		for i in range(0, 10):			
			for j in range(0, self.cluster.shape[1]):
				#Define os conjuntos de treino e teste para cada iteração. Para deixar claro, cada iteração modifica o atributo
				#que será usado como atributo classe
				cj_treino,cj_teste = self.treino_teste(j) 
				clf = copy(metodo) #Instancia uma nova rede neutal multicamadas

				#Essa parte trata os dados para ficarem de acordo com o necessário para o treinamento da MLP
				target = self.cluster[:self.porc_treino,j]	#cria uma lista com o atributo classe usado na iteração corrente
				
				clf.fit(X = cj_treino.astype(int), y = target.astype(int)) #Método que faz o treinamento da MLP.

				#Mesma intenção do trecho de código das linhas 27 e 28
				target = self.cluster[self.porc_treino:,j]
				self.resultados[j,i] = clf.score(cj_teste, target)	#Matriz resultados recebe a taxa de acerto [0, 1] da MLP treinada com o conjunto de treino


	"""Método auxiliar para separar conjunto de treino e de teste usando a coluna usada como atributo classe de acordo com a iteração corrente no método 'classificar'"""
	def treino_teste(self,j):
		#Cria o conjunto de teste/treino usando a quantidade de tuplas definidas pelo Atributo 'perc_trein' passado como parâmetro na instanciação do objeto
		treino = np.hstack((self.cluster[:self.porc_treino, :j], self.cluster[:self.porc_treino, j+1:]))
		teste = np.hstack((self.cluster[self.porc_treino:, :j],self.cluster[self.porc_treino:, j+1:]))
		return treino,teste

def classifica_bd(grupos, metodo):
	result = []

	#monta o modelo que será usado na classificação da base
	if metodo is 'MLP':
		rede = mlp(hidden_layer_sizes = (10))
	if metodo is 'TREE':
		rede = tree()

	for grupo in grupos:		
		#instancia a classe responsável pela classificação e cálculo da relevância de cada atributo
		classif = classificador(rede, grupo.as_matrix(), 60)
		result.append(classif.resultados)
		
	return result
