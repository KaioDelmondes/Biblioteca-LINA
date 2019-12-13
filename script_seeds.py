from linar.lopes import lopes_labeling as lpl
import discretizacao, agrupamento, classificacao
import utils


seeds = utils.ler_csv("tests/seeds.csv", ",")
seeds.pop('Cluster;;;;;;;;;;')
seeds.columns = ['A','P','C','LK','WK','AC','LKG']

rot = lpl(seeds)
print("Modelo Criado")
rot.set_discretizador(discretizacao.EFD, 3)
print("Discretizador Configurado")
rot.set_classificador(classificacao.MLP,(100,))
print("Classificador Configurado")
rot.set_agrupador(agrupamento.Kmeans, 3)
print("Agrupador Configurado")
rot.agrupar()
print("Agrupamento Completo")
rot.discretizar()
print("Discretização Completa")
print("GRUPOS DISCRETOS SEPARADOS")
print(rot.grupos_discretos.get_group(0).shape, rot.grupos_discretos.get_group(1).shape, rot.grupos_discretos.get_group(2).shape)

rot.avaliar_base()

print("\n",rot.resultado_rotulacao.query("RELEVÂNCIA > 80").round(2))
print("\nAVALIAÇÃO DE BASE TERMINADA")

"""
print(rot.avaliar_grupo(rot.grupos_discretos.get_group(0)))
print("AVALIAÇÃO DE GRUPO TERMINADA")
print(rot.avaliar_grupo(rot.grupos_discretos.get_group(1)))
print("AVALIAÇÃO DE GRUPO TERMINADA")
print(rot.avaliar_grupo(rot.grupos_discretos.get_group(2)))
print("AVALIAÇÃO DE GRUPO TERMINADA")

print(rot.disc_detalhes)


print(rot.avaliar_atributo(rot.grupos_discretos.get_group(0), 'ATRIBUTO_4'))
print("AVALIAÇÃO DE ATRIBUTO TERMINADA")


print(rot.avaliar_atributo(rot.grupos_discretos.get_group(0), 'ATRIBUTO_1'))
print(rot.avaliar_atributo(rot.grupos_discretos.get_group(1), 'ATRIBUTO_1'))
print(rot.avaliar_atributo(rot.grupos_discretos.get_group(2), 'ATRIBUTO_1'))
"""