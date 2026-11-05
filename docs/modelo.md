# Modelo e reprodução

## Atributos e alvo

O alvo é `preco_aluguel`: aluguel residencial mensal anunciado, em reais, sem acrescentar condomínio e IPTU. O contrato de inferência está em [`MODEL_INPUT_COLUMNS`](../src/preprocessamento.py); [`FEATURE_COLUMNS`](../src/preprocessamento.py) contém 85 atributos preparados antes da expansão de categorias e texto.

O pré-processamento normaliza cidade, bairro, rua e tipo, trata faixas numéricas incompatíveis como ausentes, extrai amenidades da descrição e gera razões e combinações de localização. Números passam por imputação pela mediana; categorias básicas usam one-hot encoding; categorias de maior cardinalidade usam target encoding suavizado. O texto é representado por TF-IDF com até 1.500 termos/unigramas e bigramas.

O target encoding das amostras de treinamento é calculado em divisões internas por grupos de imóveis. Os mapas finais são ajustados sobre o conjunto de treino da respectiva etapa e usados na transformação de novas entradas.

## Estimador

O modelo global usa XGBoost com 800 árvores, profundidade máxima 7, taxa de aprendizado 0,04, objetivo `reg:absoluteerror` e semente 42. A configuração completa está em [`build_xgboost_regressor()`](../src/modelo.py). A execução padrão utiliza CPU, com duas threads.

Antes do ajuste, o alvo é transformado em:

```text
z = log1p(aluguel / área^0,45)
peso da amostra ∝ 1 / aluguel^0,5
```

A transformação inversa restitui a escala monetária. Entradas sem área válida usam área neutra igual a 1 nessa transformação; isso preserva a execução, mas não garante uma boa estimativa.

As previsões recebem um componente de comparáveis selecionados por distância determinística entre atributos:

| Tipo | Comparáveis | Combinação final |
| --- | --- | --- |
| Casa | Até 3 casas; mediana ponderada, com ajuste de área | `exp(0,4 × log(XGBoost) + 0,6 × log(comparáveis))` |
| Apartamento | Até 5 apartamentos; mediana ponderada | `0,5 × XGBoost + 0,5 × comparáveis` |

A busca considera cidade, bairro, medidas e quantidades; apartamentos também usam rua e condomínio. A implementação está em [`src/modelo.py`](../src/modelo.py). O resultado da CLI tem piso de R$ 1 e é arredondado para inteiro; esse piso é uma proteção numérica, não uma faixa de mercado validada.

## Validação e evidência

[`train.py`](../train.py) produz previsões OOF em cinco folds. [`property_cluster_groups()`](../src/preprocessamento.py) agrupa possíveis anúncios do mesmo imóvel por regras de similaridade. [`persistent_fold_ids()`](../src/validacao.py) mantém cada grupo em um único fold, persiste a atribuição e distribui grupos novos considerando cidade e tipo quando há suporte suficiente.

Em cada fold, o modelo e seus comparáveis são ajustados somente com a parcela de treino. As métricas são calculadas sobre as previsões das parcelas retidas; depois, um modelo final é ajustado com todos os registros disponíveis.

Os valores publicados em [`reports/metrics.json`](../reports/metrics.json) são **metadados históricos do artefato original**, não resultados de um novo treinamento ou do conjunto externo da disciplina. São 3.369 linhas de origem, 3.283 após tratamento e MAPE OOF de 15,3080476%. A classificação como melhor trabalho da turma é informação dos autores e não é inferida desse MAPE.

Na preparação desta publicação, a validação OOF foi **reexecutada em CPU**, usando a mesma base e uma cópia das atribuições históricas de folds. Os arquivos originais de folds permaneceram inalterados. Essa execução ajustou os cinco modelos de validação; não fez um novo ajuste final sobre todos os registros nem substituiu o artefato de inferência.

O resultado foi MAPE de **15,2233181%**, MAE de R$ 670,28 e R² de 0,8160871. As métricas completas, por fold e tipo, estão em [`reports/oof-metrics.json`](../reports/oof-metrics.json). Os valores dessa execução e os históricos são mantidos separadamente; a pequena diferença não é atribuída a uma causa confirmada.

| Tipo de imóvel | Registros | MAPE OOF reexecutado |
| --- | ---: | ---: |
| Apartamento | 2.655 | 13,93% |
| Casa | 628 | 20,70% |

O [gráfico de previsão versus anúncio](assets/predicoes-oof.png) usa os [resultados anonimizados](../reports/oof-predictions.csv), que contêm somente valor anunciado, previsão, tipo e fold. Ele não mostra previsões do modelo final sobre o próprio treino.

A reprodução exata exige a mesma base e as atribuições de folds originais, que não integram a versão pública. Uma coleta nova não deve ser apresentada como reprodução exata do resultado histórico.

## Treinar com outra base

Depois de instalar [`requirements.txt`](../requirements.txt), forneça um CSV próprio com o contrato de inferência, `preco_aluguel` e metadados de qualidade: `fonte`, `url`, `data_coleta` e evidência de periodicidade mensal. Consulte os filtros em [`prepare_model_frame()`](../src/preprocessamento.py) e as regras em [`src/data_quality.py`](../src/data_quality.py). Linhas sem área válida, data de coleta ou evidência aceita de aluguel mensal podem ser excluídas.

```bash
python train.py \
  --dataset dados/processados/minha_base.csv \
  --folds dados/validacao/meus_folds.csv \
  --predictions-out dados/validacao/minhas_predicoes.csv \
  --results-out dados/validacao/meus_resultados.csv \
  --model-out meu_modelo.joblib
```

O script mantém o teto acadêmico de MAPE OOF de **15,6%** e interrompe a gravação do modelo quando ele é ultrapassado. Esse teto é um critério do experimento original; para outro conjunto de dados, sua adequação precisa ser reavaliada. A base deve conter grupos suficientes para as cinco divisões e exemplos dos dois tipos de imóvel usados pelos comparáveis.

## Artefato publicado

O arquivo [`modelo_aluguel.joblib`](../modelo_aluguel.joblib) contém o estimador, os transformadores, estatísticas aprendidas e referências de comparáveis. As referências conservam somente os atributos necessários à previsão e seus aluguéis; não incluem descrições brutas, URLs ou contatos. Portanto, o artefato contém informação derivada do treino e não deve ser tratado como um arquivo sem dados de referência.

Na preparação pública foram removidas colunas desnecessárias das referências. A equivalência com o modelo original foi conferida nas 3.283 entradas de referência: diferença absoluta máxima de previsão **0,0**. Esse resultado está registrado em `publication` no JSON de métricas.

SHA-256 do artefato:

```text
6ebc31f1c65aea989232bf7d4bfebbad9c0045e41a85b99c2d78782511ab24ef
```

Use as versões fixadas das dependências. Arquivos Joblib/Pickle podem executar código ao serem carregados: carregue apenas artefatos de origem confiável.
