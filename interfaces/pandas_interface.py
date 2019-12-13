"""
OBSERVAÇÕES:
	O parâmetro labels nos métodos cut e qcut simplificam o retorno do método, mas levam aperda de informação. Informações como intervalos de valor de cada faixa seria útil
"""

from pandas import qcut, cut

def EWD_to_pandas(array,qt_faixas):
	try:
		data, intervalos = cut(array, bins=qt_faixas, labels=False, retbins=True)
		return data, intervalos
	except Exception as e:
		raise e

def EFD_to_pandas(array,qt_faixas):
	try:
		data, intervalos = qcut(array, q=qt_faixas, labels=False, retbins=True)
		return data, intervalos
	except Exception as e:
		raise e