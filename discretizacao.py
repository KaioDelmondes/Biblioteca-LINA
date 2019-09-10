import interfaces.pandas_interface as pandas_interface
import numpy

def EWD(array, qt_faixas):
	return pandas_interface.EWD_to_pandas(array, qt_faixas)

def EFD(array, qt_faixas):
	return pandas_interface.EFD_to_pandas(array, qt_faixas)

def EWD_for_matrix(matrix, qt_faixas):
	try:
		qt_columns = matrix.shape[1]
	except IndexError:
		qt_columns = 1

	is_faixas_valid(qt_faixas, qt_columns)
	n_matrix = numpy.asarray(matrix)

	if qt_columns == 1:
		if isinstance(qt_faixas, list):
			if len(qt_faixas) == 1:
				return EWD(n_matrix, qt_faixas[0])
		return EWD(n_matrix, qt_faixas)

	discrete_matrix = numpy.zeros(n_matrix.shape)
	
	for i in range(qt_columns):
		if isinstance(qt_faixas, list):
			if len(qt_faixas) == 1:
				discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas[0])
			else:
				discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas[i])
		else:
			discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas)
	
	return discrete_matrix

def EFD_for_matrix(matrix, qt_faixas):
	try:
		qt_columns = matrix.shape[1]
	except IndexError:
		qt_columns = 1

	is_faixas_valid(qt_faixas, qt_columns)
	n_matrix = numpy.asarray(matrix)

	if qt_columns == 1:
		if isinstance(qt_faixas, list):
			if len(qt_faixas) == 1:
				return EWD(n_matrix, qt_faixas[0])
		return EWD(n_matrix, qt_faixas)

	discrete_matrix = numpy.zeros(n_matrix.shape)
	
	for i in range(qt_columns):
		if isinstance(qt_faixas, list):
			if len(qt_faixas) == 1:
				discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas[0])
			else:
				discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas[i])
		else:
			discrete_matrix[:,i]=EWD(n_matrix[:,i], qt_faixas)
	
	return discrete_matrix

def is_faixas_valid(faixas, qt_columns):
	if isinstance(faixas, int) and faixas > 0:
		return True
	elif isinstance(faixas,list) and all(isinstance(n,int) and n > 0 for n in faixas):
		if len(faixas) == qt_columns or len(faixas) == 1:
			return True
		else:
			raise ValueError("A lista deve ser inteira e positiva, com um único elemento ou com a mesma quantidade de colunas da matriz")
	else:
		raise TypeError("A entrada desse argumento deve ser uma lista de inteiros positivos com tamanho igual à quantidade de colunas da matriz, ou um inteiro que será usada para todas as colunas da matriz")