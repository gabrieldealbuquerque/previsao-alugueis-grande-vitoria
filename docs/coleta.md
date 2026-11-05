# Coleta e qualidade dos dados

## O que acompanha a versão pública

[`scrapers/`](../scrapers/) preserva os coletores do trabalho acadêmico para documentar a obtenção e a consolidação dos anúncios. A base original, anúncios brutos, arquivos de avaliação da disciplina e credenciais não são distribuídos. [`examples/entrada.csv`](../examples/entrada.csv) contém somente três imóveis fictícios.

O modelo treinado inclui atributos mínimos e valores de aluguel dos imóveis usados como comparáveis, além de representações aprendidas. Veja [o conteúdo do artefato](modelo.md#artefato-publicado).

| Módulo | Fonte prevista no código |
| --- | --- |
| `scrapers.scraper` | Chaves na Mão |
| `scrapers.scraper_mgf` | MGF Imóveis |
| `scrapers.scraper_netimoveis` | Netimóveis |
| `scrapers.scraper_itamar` | Itamar Imóveis |
| `scrapers.scraper_tiosam` | Tio Sam Imóveis |
| `scrapers.scraper_portais` | OLX, ZAP Imóveis, Viva Real e Imovelweb |

Esses módulos refletem a estrutura observada nos portais durante o projeto. **Não foi executada uma nova coleta para esta publicação**, e não há garantia de que seletores, respostas ou disponibilidade permaneçam compatíveis. A inclusão de um coletor não prova que todos os seus registros compõem a base final.

## Preparar uma coleta própria

Antes de executar, confira os termos de uso e as permissões da fonte, limite requisições e evite coletar contatos ou outros dados pessoais desnecessários. Não use o código para contornar bloqueios de acesso.

As dependências de coleta são separadas das de inferência:

```bash
python -m pip install -r requirements-coleta.txt
PLAYWRIGHT_BROWSERS_PATH=.venv/playwright-browsers python -m playwright install chromium
python -m scrapers.scraper_portais --help
```

O comando de ajuda não inicia uma coleta. A execução efetiva pode gravar arquivos em `dados/brutos/` e `dados/processados/`, definidos em [`scrapers/paths.py`](../scrapers/paths.py). Essa pasta é ignorada pelo Git. O treinamento não transforma automaticamente a saída de qualquer coletor em uma base final validada.

## Critérios de qualidade

O pipeline busca distinguir aluguel residencial mensal de venda, temporada, diária, compartilhamento e uso comercial. Entre as regras implementadas:

- Normalização de cidades da Grande Vitória, tipos e campos numéricos.
- Classificação de periodicidade com evidências explícitas; fontes têm regras próprias em [`src/data_quality.py`](../src/data_quality.py).
- Exclusão de pacotes que incluem encargos quando o aluguel isolado não é identificável.
- Tratamento de faixas incompatíveis de área, quantidades e encargos como ausentes.
- Deduplicação por fonte/identificador e tratamento de anúncios sindicados entre ZAP e Viva Real.
- Agrupamento de imóveis semelhantes para manter possíveis duplicatas no mesmo fold de validação.

No treinamento, é necessário informar o alvo positivo, área válida e data de coleta. Registros MGF exigem confirmação de anúncio atual conforme o contrato do pré-processamento. As regras são heurísticas e precisam de inspeção amostral em qualquer base nova.

## Limites de interpretação

Anúncios são uma amostra do mercado disponível nas fontes consultadas. Podem conter erros, duplicatas residuais, valores desatualizados ou características incompletas. O preço anunciado não comprova uma transação, e a distribuição de cidades, bairros e faixas de preço pode ser desigual.

Para evoluir o projeto, registre a proveniência e a data dos dados, revise amostras e preserve uma separação temporal para avaliar anúncios futuros. Métricas de uma coleta nova devem ser reportadas com seu próprio período, escopo e protocolo de validação.
