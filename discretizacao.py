import interfaces.pandas_interface as pandas_interface

def EWD(array, qt_faixas):
	return pandas_interface.EWD_to_pandas(array, qt_faixas)

def EFD(array, qt_faixas):
	return pandas_interface.EFD_to_pandas(array, qt_faixas)