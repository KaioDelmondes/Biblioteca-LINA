import interfaces.pandas_interface as pandas_interface
import numpy

class generalizacao_discretizadores:
	def __init__(self, data, faixas_discretizacao):
		self.data = numpy.asarray(data)
		self.faixas_discretizacao = faixas_discretizacao
		self.is_faixas_valid()
		self.discrete_data = numpy.zeros(self.data.shape)

	def discretizar(self):
		if self.qt_columns == 1:
			if isinstance(self.faixas_discretizacao, list):
				if len(self.faixas_discretizacao) == 1:
					self.discrete_data =  self.metodo_de_discretizacao(self.data, self.faixas_discretizacao[0])
			else:
				self.discrete_data = self.metodo_de_discretizacao(self.data, self.faixas_discretizacao)

		else:
			for i in range(self.qt_columns):
				if isinstance(self.faixas_discretizacao, list):
					if len(self.faixas_discretizacao) == 1:
						self.discrete_data[:,i]= self.metodo_de_discretizacao(self.data[:,i], self.faixas_discretizacao[0])
					else:
						self.discrete_data[:,i]= self.metodo_de_discretizacao(self.data[:,i], self.faixas_discretizacao[i])
				else:
					self.discrete_data[:,i]= self.metodo_de_discretizacao(self.data[:,i], self.faixas_discretizacao)

	def is_faixas_valid(self):
		try:
			self.qt_columns = self.data.shape[1]
		except IndexError:
			self.qt_columns = 1

		if isinstance(self.faixas_discretizacao, int) and self.faixas_discretizacao > 0:
			return True
		elif isinstance(self.faixas_discretizacao,list) and all(isinstance(n,int) and n > 0 for n in self.faixas_discretizacao):
			if len(self.faixas_discretizacao) == self.qt_columns or len(self.faixas_discretizacao) == 1:
				return True
			else:
				raise ValueError("A lista deve ser inteira e positiva, com um único elemento ou com a mesma quantidade de colunas da matriz")
		else:
			raise TypeError("A entrada desse argumento deve ser uma lista de inteiros positivos com tamanho igual à quantidade de colunas da matriz, ou um inteiro que será usada para todas as colunas da matriz")

class EWD(generalizacao_discretizadores):
	def __init__(self, data, faixas_discretizacao):
		super().__init__(data, faixas_discretizacao)

	def metodo_de_discretizacao(self, array, intervalo):
		return pandas_interface.EWD_to_pandas(array, intervalo)

class EFD(generalizacao_discretizadores):
	"""docstring for EFD"""
	def __init__(self, data, faixas_discretizacao):
		super().__init__(data, faixas_discretizacao)
	
	def metodo_de_discretizacao(self, array, intervalo):
			return pandas_interface.EFD_to_pandas(array, intervalo)