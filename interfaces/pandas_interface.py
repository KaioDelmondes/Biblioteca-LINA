"""
OBSERVAÇÕES:
	O parâmetro labels nos métodos cut e qcut simplificam o retorno do método, mas levam aperda de informação. Informações como intervalos de valor de cada faixa seria útil
"""

import pandas

def EWD_to_pandas(array,qt_faixas):
	try:
		atributo_disc = pandas.cut(array, bins=qt_faixas, labels=False)
		return atributo_disc
	except Exception as e:
		raise e

def EFD_to_pandas(array,qt_faixas):
	try:
		atributo_disc = pandas.qcut(array, q=qt_faixas, labels=False)
		return atributo_disc
	except Exception as e:
		raise e