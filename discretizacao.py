import interfaces.pandas_interface as pandas_interface
import numpy
from pandas import DataFrame

class generalizacao_discretizadores:
	def __init__(self, data, faixas_discretizacao):
		self.data = DataFrame(data)
		self.faixas_discretizacao = faixas_discretizacao
		self.is_faixas_valid()
		self.discrete_data = DataFrame(numpy.zeros(self.data.shape), columns = self.data.columns, index = self.data.index)
		self.disc_detalhes = None

	def discretizar(self):
		if self.qt_columns == 1:

			if isinstance(self.faixas_discretizacao, list):
				self.discrete_data, intervalos =  self.metodo_de_discretizacao(self.data.values, self.faixas_discretizacao[0])
			else:
				self.discrete_data, intervalos = self.metodo_de_discretizacao(self.data.values, self.faixas_discretizacao)


		else:
			intervalos = []

			for i in range(self.qt_columns):
				if isinstance(self.faixas_discretizacao, list):
					self.discrete_data.values[:,i], intervalo = self.metodo_de_discretizacao(self.data.values[:,i], self.faixas_discretizacao[i])
					intervalos.append(intervalo)
				else:
					self.discrete_data.values[:,i], intervalo = self.metodo_de_discretizacao(self.data.values[:,i], self.faixas_discretizacao)
					intervalos.append(intervalo)

		self.detalha_disc(intervalos)

	def is_faixas_valid(self):
		try:
			self.qt_columns = self.data.shape[1]
		except IndexError:
			self.qt_columns = 1

		if isinstance(self.faixas_discretizacao, int) and self.faixas_discretizacao > 0:
			return True
		elif isinstance(self.faixas_discretizacao,list) and all(isinstance(n,int) and n > 0 for n in self.faixas_discretizacao):
			if len(self.faixas_discretizacao) == self.qt_columns:
				return True
			else:
				raise ValueError("A lista deve ser inteira e positiva, com a mesma quantidade de colunas da matriz")
		else:
			raise TypeError("A entrada desse argumento deve ser uma lista de inteiros positivos com tamanho igual à quantidade de colunas da matriz, ou um inteiro que será usada para todas as colunas da matriz")

	def detalha_disc(self, intervalos):
		rows_labels, col_labels = self.get_data_labels(intervalos)
		self.disc_detalhes = DataFrame(intervalos, index=rows_labels, columns=col_labels) if isinstance(intervalos,list) else DataFrame([intervalos], index=rows_labels, columns=col_labels)

	def get_data_labels(self, inter):
		pontos_de_corte = max(self.faixas_discretizacao) if isinstance(self.faixas_discretizacao, list) else self.faixas_discretizacao
		col_labels = ["P_C %s"%x for x in range(1, pontos_de_corte+2)]
		rows_labels = self.data.columns
		return rows_labels,col_labels


class EWD(generalizacao_discretizadores):
	def __init__(self, data, faixas_discretizacao):
		super().__init__(data, faixas_discretizacao)
		self.metodo_de_discretizacao = pandas_interface.EWD_to_pandas

class EFD(generalizacao_discretizadores):
	"""docstring for EFD"""
	def __init__(self, data, faixas_discretizacao):
		super().__init__(data, faixas_discretizacao)
		self.metodo_de_discretizacao = pandas_interface.EFD_to_pandas