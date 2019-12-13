from interfaces.scikitlearn_interface import MLP_sklearn
import numpy

class MLP:
	"""docstring for MLP"""
	def __init__(self, camadas_ocultas=()):
		self.modelo = MLP_sklearn(camadas_ocultas)

	def treinar(self, entradas, attbr_classe):
		self.modelo.treinar(entradas, attbr_classe)
	
	def estimar(self, attbr):
		attbr = numpy.asarray(attbr)
		if attbr.ndim == 2:
			return self.modelo.estimar_n(attbr)
		else:
			return self.modelo.estimar_um(attbr)

	def precisao_do_modelo(self, entradas, attbr_classe):
		return self.modelo.precisao(entradas, attbr_classe)