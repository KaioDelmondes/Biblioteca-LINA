import numpy as np
from pandas import DataFrame, MultiIndex
import warnings
from collections import Counter
warnings.simplefilter('ignore')

class lopes_labeling:
	def __init__(self, data, holdout_value = 10):
		self.db_continuo = data
		self.db_discreto = None
		self.db_agrupado = None
		self.grupos_discretos = None
		self._metodo_agrupar, self._metodo_discretizar, self._metodo_classificar = None, None, None
		self._metodo_agrupar_params, self._metodo_discretizar_params, self._metodo_classificar_params = None, None, None
		self.holdout_value = holdout_value
	
	def agrupar(self):
		self.db_agrupado = self._metodo_agrupar(self.db_continuo, *self._metodo_agrupar_params)
	
	def discretizar(self):
			discretizador = self._metodo_discretizar(self.db_continuo, *self._metodo_discretizar_params)
			discretizador.discretizar()
			self.db_discreto, self.disc_detalhes = discretizador.discrete_data, discretizador.disc_detalhes
			self._dividir_db_discretizado()
	
	def set_agrupador(self, metodo, *params):
		self._metodo_agrupar = metodo
		self._metodo_agrupar_params = params

	def set_discretizador(self, metodo, *params):
		self._metodo_discretizar = metodo
		self._metodo_discretizar_params = params

	def set_classificador(self, metodo, *params, perc_treino=80, perc_teste=100):
		self._metodo_classificar = metodo
		self._metodo_classificar_params = params
		self._perc_treino, self._perc_teste = perc_treino, perc_teste
	
	def _get_treino_teste(self, grupo, atributo_classe):
		frac_treino = self._perc_treino/100
		frac_teste = self._perc_teste/100
		cj_treino = grupo.sample(frac=frac_treino)
		cj_teste = grupo.drop(cj_treino.index).sample(frac=frac_teste)
		return cj_treino, cj_teste

	def _dividir_db_discretizado(self, col_referencia = 'col_cluster'):
		col_cluster = self.db_agrupado[col_referencia]
		self.grupos_discretos = self.db_discreto.groupby(col_cluster)

	def _monta_rotulo(self, grupo, atributo):
		valor_discreto, ocorrencia = Counter(grupo[atributo]).most_common(1).pop()
		erros = len(grupo) - ocorrencia
		acerto = 100 - round((erros/len(grupo))*100,2)
		infor_adicionais = {'ERROS/TOTAL': f'{erros}/{len(grupo)}', 'ACERTO(%)': acerto}
		
		query = "P_C [%s,%s]$"%(int(valor_discreto+1), int(valor_discreto+2))
		pontos_de_corte = self.disc_detalhes.filter(regex=query, axis=1).loc[[atributo]]
		pontos_de_corte.columns = ['LIM INF', 'LIM SUP']

		return infor_adicionais, pontos_de_corte
	
	def avaliar_atributo(self, grupo, atributo):
		cj_treino, cj_teste = self._get_treino_teste(grupo, atributo)
		
		if cj_treino.empty or cj_teste.empty:
			relevancia_df = DataFrame([np.nan], columns=['RELEVÂNCIA'], index=[atributo])
			return relevancia_df

		relevancia_array = np.zeros(shape = self.holdout_value)
		
		for hvl in range(self.holdout_value):
			classificador = self._metodo_classificar(*self._metodo_classificar_params)

			valores_entrada_treino = cj_treino.drop(atributo, axis=1)
			valores_classe_treino = cj_treino[atributo]

			valores_entrada_teste = cj_teste.drop(atributo, axis=1)
			valores_classe_teste = cj_teste[atributo]
			
			classificador.treinar(valores_entrada_treino,valores_classe_treino)
			relevancia_array[hvl] = classificador.precisao_do_modelo(valores_entrada_teste, valores_classe_teste)

		val_relevancia = relevancia_array.mean()
		relevancia_df = DataFrame(val_relevancia, columns=['RELEVÂNCIA'], index=[atributo])

		infor_adicionais, pontos_de_corte = self._monta_rotulo(grupo = grupo, atributo = atributo)

		relevancia_df = relevancia_df.assign(**pontos_de_corte,**infor_adicionais)
		return relevancia_df

	def avaliar_grupo(self, grupo):
		cluster = grupo if isinstance(grupo, DataFrame) else self.grupos_discretos.get_group(grupo)

		relevancia_dos_atributos = DataFrame()

		for atributo in cluster.columns:
			relevancia_dos_atributos = relevancia_dos_atributos.append(self.avaliar_atributo(cluster, atributo))

		return relevancia_dos_atributos.sort_values(by=['RELEVÂNCIA'], ascending=False)

	def avaliar_base(self):
		self.resultado_rotulacao = DataFrame()
		grupos_labels = []

		for chave_grupo, grupo in self.grupos_discretos:
			relevancia_no_grupo = self.avaliar_grupo(grupo)
			
			grupos_labels.extend([chave_grupo]*len(relevancia_no_grupo.index))

			self.resultado_rotulacao = self.resultado_rotulacao.append(relevancia_no_grupo)

		array_labels = [grupos_labels, self.resultado_rotulacao.index]
		new_index = MultiIndex.from_arrays(array_labels, names=('GRUPO', 'ATRIBUTOS'))

		self.resultado_rotulacao.index = new_index

		return self.resultado_rotulacao
	
	def rotulo(self, V=10):
		rotulos = []
		for chave_grupo, _ in self.grupos_discretos:
			grupo_rot = []
			limiar = (100-V)/100 * self.resultado_rotulacao.loc[chave_grupo]['RELEVÂNCIA'].max()
			validos = self.resultado_rotulacao.loc[chave_grupo][['RELEVÂNCIA', 'LIM INF', 'LIM SUP']].query(f'RELEVÂNCIA >= {limiar}').round(2)
			for atributo, valores in validos.iterrows():
				grupo_rot.append((atributo, valores['LIM INF'], valores['LIM SUP']))
			rotulos.append(grupo_rot)
		return rotulos
		