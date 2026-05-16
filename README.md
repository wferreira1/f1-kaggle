# f1-kaggle

Projeto desenvolvido para a competição **Kaggle Playground Series - Season 6, Episode 5: _Predicting F1 Pit Stops_**.

O objetivo da competição é prever a probabilidade de um piloto de Fórmula 1 realizar um pit stop na próxima volta, representada pela variável-alvo `PitNextLap`. O problema é de classificação binária e a submissão final deve conter uma probabilidade para cada linha do conjunto de teste.

## Estado atual do projeto

O repositório já possui um pipeline inicial funcional. No momento, ele contém:

- os arquivos de entrada da competição baixados localmente;
- uma estrutura separada para código-fonte, dados de entrada, modelos e saídas;
- um script para download dos dados via Kaggle API;
- engenharia de atributos em `src/features.py`;
- treino de dois modelos baseline em `src/train.py`;
- geração de submissões em `src/predict.py`;
- artefatos de modelo salvos em `models/` e submissões exportadas para `data/out/`.

## Estrutura do repositório

```text
f1-kaggle/
├── data/
│   ├── in/
│   │   ├── playground-series-s6e5.zip
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── out/
│       └── submission.csv
├── models/
├── src/
│   ├── download_data.sh
│   ├── features.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── main.py
├── pyproject.toml
└── README.md
```

## Dados

Os dados da competição estão armazenados em `data/in/`.

### Arquivos principais

| Arquivo | Descrição |
| --- | --- |
| `train.csv` | Conjunto de treino com 439.140 linhas e a variável-alvo `PitNextLap`. |
| `test.csv` | Conjunto de teste com 188.165 linhas, sem a variável-alvo. |
| `sample_submission.csv` | Exemplo do formato esperado para submissão. |
| `submission.csv` | Submissão atualmente disponível no projeto. |

### Variáveis disponíveis

O conjunto de treino contém as seguintes colunas:

- Identificação: `id`
- Contexto da corrida: `Driver`, `Race`, `Year`
- Estratégia e stint: `Compound`, `PitStop`, `Stint`, `TyreLife`
- Estado da volta: `LapNumber`, `Position`, `LapTime (s)`
- Variáveis derivadas: `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`
- Alvo: `PitNextLap`

## Fluxo de trabalho atual

O pipeline atual está organizado da seguinte forma:

1. **Download dos dados** com `src/download_data.sh`.
2. **Engenharia de atributos** em `src/features.py`.
3. **Treinamento e validação** em `src/train.py`.
4. **Geração de previsões** em `src/predict.py`.
5. **Funções compartilhadas** em `src/utils.py`.
6. **Armazenamento de modelos** em `models/`.
7. **Exportação de submissões** em `data/out/`.

## Como baixar os dados

Com a Kaggle API configurada na máquina, execute:

```bash
bash src/download_data.sh
```

O script baixa o pacote da competição e extrai os arquivos dentro de `data/in/`.

## Como treinar os modelos

Para validar rapidamente o funcionamento do pipeline em ambiente local, use uma amostra aleatória pequena:

```bash
uv sync
uv run python src/train.py --sample-size 1000
```

Esse comando:

- divide o treino em treino/validação;
- ajusta o pipeline de features sem vazar informação da validação;
- remove `id` do conjunto de variáveis preditoras;
- treina versões leves de `RandomForestClassifier` e `ExplainableBoostingClassifier`;
- mede ROC AUC em validação;
- reajusta os modelos com 100% dos dados de treino;
- salva `feature_artifacts.pkl`, `random_forest.pkl` e `ebm.pkl` em `models/`.

Para o treino completo, execute o mesmo script **sem** `--sample-size`, preferencialmente no Kaggle:

```bash
python src/train.py
```

Sem `--sample-size`, o script usa a configuração completa dos modelos.

## Como gerar uma submissão

Por padrão, a submissão usa o modelo EBM:

```bash
uv run python src/predict.py
```

Para testar apenas o fluxo de inferência com poucas linhas:

```bash
uv run python src/predict.py --sample-size 1000
```

Por padrão, a amostragem usa `seed=42`. Se quiser variar a amostra:

```bash
uv run python src/train.py --sample-size 1000 --sample-seed 123
```

Para usar o Random Forest:

```bash
uv run python src/predict.py --model random_forest
```

O arquivo final é salvo em `data/out/submission.csv`.

## Execução no Kaggle com Kaggle CLI

O fluxo recomendado é publicar o código do projeto como um **Dataset privado** no Kaggle e usar um **Kernel script** pequeno para executar o treino completo na infraestrutura deles.

### 1. Configurar o Kaggle CLI

Instale o CLI e configure suas credenciais em `~/.kaggle/kaggle.json`.

### 2. Publicar o código como Dataset

Para criar o dataset pela primeira vez:

```bash
bash scripts/deploy_kaggle_dataset.sh create SEU_USUARIO/f1-kaggle-code "F1 Kaggle Project Code"
```

Para publicar novas versões depois:

```bash
bash scripts/deploy_kaggle_dataset.sh version SEU_USUARIO/f1-kaggle-code "F1 Kaggle Project Code" "Atualiza pipeline"
```

O script empacota apenas os arquivos necessários do projeto (`src/`, `README.md`, `pyproject.toml`, `uv.lock` e `main.py`) e ignora dados locais, modelos e caches.

### 3. Preparar o Kernel remoto

Copie o template:

```bash
cp kaggle/kernel/kernel-metadata.json.example kaggle/kernel/kernel-metadata.json
```

Depois, edite `kaggle/kernel/kernel-metadata.json` e substitua:

- `SEU_USUARIO/f1-pit-stop-training`
- `SEU_USUARIO/f1-kaggle-code`

### 4. Enviar e executar o Kernel no Kaggle

```bash
bash scripts/push_kaggle_kernel.sh
```

O arquivo `kaggle/kernel/run_training.py` usa:

- o dataset de código publicado por você;
- os dados da competição `playground-series-s6e5`;
- `/kaggle/working/` para salvar modelos e `submission.csv`.

## Próximos passos recomendados

- comparar estratégias de validação mais próximas do cenário temporal da competição;
- explorar novos modelos e tuning de hiperparâmetros;
- registrar resultados de experimentos e versões de submissão;
- decidir se `main.py` deve virar um ponto único de orquestração do pipeline.

## Competição

- Competição: **Predicting F1 Pit Stops**
- Série: **Playground Series - Season 6, Episode 5**
- Métrica usada no ranking: **ROC AUC**
- Encerramento previsto: **31 de maio de 2026, 23:59 UTC**
