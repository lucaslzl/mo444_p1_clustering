## MO444/MC886 - Aprendizado de Máquina e Reconhecimento de Padrões

Esse trabalho foi feito pelos seguintes membros:

- Lucas Zanco Ladeira - 188951
- Rafael Scherer - 204990 

O código original deste projeto está disponível no [repositório do Github](https://github.com/lucaslzl/mo444_p1_clustering). 


## Modelos de Agrupamento e Redução de Dimensionalidade

Neste trabalho foi necessário implementar dois modelos de agrupamento e utilizar o algoritmo <b>Principal Component Analysis (PCA)</b> da biblioteca scikit-learn para a tarefa de redução de dimensionalidade. Os modelos implementados compreendem o <b>KMeans</b> e o <b>DBScan</b>. De forma resumida, no primeiro caso são selecionados centróides de clusters e analisados os clusters formados por esses centróides. O algoritmo é iterado um determinado número de vezes e clusters são encontrados na qual a distância entre os membros dos clusters e os centróides seja mínima dentro das iterações. No segundo caso, é calculada a distância entre todos os registros para encontrar quais formam clusters considerando uma distância máxima e uma vizinhança mínima. Os registros são classificados como <i>outlier</i>, borda, e central. Sendo assim, é um algoritmo interessante para identificar <i>outliers</i> dentro de conjuntos de dados.
