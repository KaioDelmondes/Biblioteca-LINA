import pandas as pd
import numpy as np

# Classe de discretizacao de dados 
# Recebe como parâmetros a lista de atributos (attr_list) a serem discretizados e o número de faixas desejado (num_bins)
# Retorna a lista de atributos discretizada (discr_list)]

class discretization (object):
	
	def __init__ (self, attr_list, num_bins):
		self.attr_list = attr_list
		self.num_bins = num_bins		

	#Testa se a lista de valores recebida é discreta, retornando True, ou contínua, retornando False.
	def teste (self):
		unics = sorted(set(self.attr_list))
		if len(unics) < self.num_bins: 
			return True
		else: 
			return False 

	# Discretiza os dados pelo critério de larguras iguais. Retorna faixas de valores com aproximadamente o mesmo tamanho.
	def EWD (self):
		if not self.teste():
			discr_attr = pd.cut(self.attr_list, bins = self.num_bins, labels = False, retbins= True)
			return discr_attr
		else :
			return self.attr_list
			
	# Disretiza os dados pelo critério de frequências iguais.Retorna faixas de valores com aproximadamente o mesmo número de dados.
	def EFD (self):
		if not self.teste():
			discr_attr = pd.qcut(self.attr_list, self.num_bins, labels = False, retbins = True, duplicates = 'drop')
			return discr_attr
		else: 
			return self.attr_list

def discretize (db, faixa_de_valores, metodo):
	ddb = []
	info = []
	
	# Para cada atributo, chama a classe de discretização, passando como
	# parâmetro a coluna do atributo e o número de faixas desejado
	for j in range(0, len(faixa_de_valores)):
		if metodo is "EFD":
			disc_attb = discretization(db[:,j], faixa_de_valores[j]).EFD()
		elif metodo is "EWD":
			disc_attb = discretization(db[:,j], faixa_de_valores[j]).EWD()
		ddb.append(disc_attb[0])
		info.append(disc_attb[1])
	ddb = np.asarray(ddb, dtype = 'int32')
	return ddb.T, info

def discretize_db(db, faixa_de_valores, metodo):
	
	col = db.shape[1]-1                             # shape retorna uma tupla com o número de linhas e colunas 
	data = db.drop(db.columns[col], axis=1)		    # data copia os valores da base de dados ignorando o atributo cluster
	
	#Copia a base de dados original
	ddb = db.copy()
	
	# Chama o método de discretização
	ddb_, info = discretize(data.get_values(), faixa_de_valores, metodo)
	
	# Sobrescreve ddb com os atributos discretizados
	for x in range (0,col):
	   ddb.loc[:,ddb.columns[x]] = [y[x] for y in ddb_]
	
	#Retorna a base de dados discretizada, as informações da discretização e um frame de cada grupo
	return ddb, info