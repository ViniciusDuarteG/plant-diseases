# Plant Diseases AI

Classificação de imagens de plantas em 38 classes de espécie e condição
(doença, praga ou planta saudável), usando `yolo26n-cls.pt` e Ultralytics.
O modelo atribui uma classe à imagem inteira; não localiza lesões.

## Treinamento no Google Colab

O fluxo está em [notebooks/treino_colab.ipynb](notebooks/treino_colab.ipynb).
[Abrir no Google Colab](https://colab.research.google.com/github/ViniciusDuarteG/plant-diseases/blob/main/notebooks/treino_colab.ipynb).
No Colab, use **Arquivo → Fazer upload de notebook** para abrir esse arquivo.
Também é possível abri-lo pela opção **GitHub** do Colab, usando o repositório
`ViniciusDuarteG/plant-diseases`.

O notebook contém instalação das dependências, conexão ao Drive, extração dos
dados, conferência da GPU e das classes, treinamento, avaliação no teste e
download do melhor modelo. Ele está configurado para 30 épocas, com parada
antecipada após 10 épocas sem melhora na validação.

**Se já existe um treinamento em andamento, deixe-o terminar. Salvar este
notebook no repositório não exige reiniciar o treinamento.**

### Preparar e enviar os dados

Coloque as imagens originais em `dataset/raw/<classe>/` e execute:

```bash
.venv/bin/python training/prepare_dataset.py
```

Esse script preserva `raw`, exclui cópias idênticas da seleção e recria
`train`, `val` e `test`. Não o execute enquanto um treino local estiver lendo
esses conjuntos.

Se os dados já foram preparados e conferidos, basta compactá-los, a partir
da raiz do projeto:

```bash
tar -czf plant-diseases-colab.tar.gz \
  training tests requirements.txt \
  dataset/train dataset/val dataset/test
```

Envie o arquivo para `Meu Drive/plant-diseases-ai/`. No Colab, selecione
**Ambiente de execução → Alterar tipo de ambiente de execução → GPU** e
execute as células na ordem.

As imagens são extraídas no disco temporário do Colab para leitura durante
o treino. Os pesos e resultados são gravados no Drive:

```text
Meu Drive/plant-diseases-ai/
├── plant-diseases-colab.tar.gz
├── runs/
│   └── plant-diseases-v1/
│       ├── args.yaml
│       ├── results.csv
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── evaluations/
```

O Colab pode encerrar sessões, e a disponibilidade de GPU não é garantida.
Se a sessão for recriada, monte o Drive e prepare novamente os dados locais.
Para avaliar um treino já concluído, pule a célula de treinamento e use o
`RUN_NAME` correspondente. A célula de treino bloqueia a reutilização de uma
pasta de execução existente para evitar reiniciar um experimento por engano.

Referências: [Colab](https://research.google.com/colaboratory/faq.html) e
[classificação com Ultralytics](https://docs.ultralytics.com/tasks/classify/).

## Estado registrado

Resultado do teste inicial de 3 épocas, conforme log fornecido em 24/09/2026:

| Item | Valor |
| --- | --- |
| GPU | Tesla T4 |
| Imagens de treino | 43.412 |
| Imagens de validação | 5.415 |
| Imagens de teste reservadas | 5.457 |
| Classes | 38 |
| Duração | aproximadamente 15 minutos |
| Acurácia top-1 na validação | aproximadamente 99% |
| Execução | `plant-diseases-smoke` |

O log registra Ultralytics `8.4.161`, PyTorch `2.11.0+cu128` e Python
`3.13.15`. O notebook fixa a versão do Ultralytics e registra as versões
efetivas do ambiente; utiliza o PyTorch com CUDA disponibilizado pelo Colab.

**Os 99% são da validação do treino curto.** Ainda não há resultado do treino
completo ou do conjunto de teste registrado neste repositório. Essa métrica
também não mede o desempenho em fotos de campo fora deste dataset.

## Desenvolvimento local

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Estrutura principal:

```text
training/prepare_dataset.py  # Preparação, deduplicação e divisão dos dados
training/train.py            # Treino local curto (3 épocas)
tests/predict_image.py       # Predição manual de uma imagem
notebooks/treino_colab.ipynb # Treino e avaliação no Colab
```

Para testar localmente com o script de predição atual, baixe o `best.pt` da
execução desejada e salve uma cópia em `models/tomato-disease-v1.pt`, que é o
caminho esperado por esse script. Apesar desse nome, o modelo cobre as 38
classes de várias plantas.

```bash
.venv/bin/python tests/predict_image.py /caminho/para/imagem.jpg
```

O script de predição imprime a classe e a confiança; não é um teste
automatizado nem rejeita imagens fora do escopo.

## O que versionar

Versione código, notebook sem saídas, documentação e pequenos relatórios
de métricas. O `.gitignore` exclui imagens, pesos, arquivos compactados,
ambiente virtual e saídas de treinamento. Mantenha os dados e modelos no
Drive. O notebook salvo aqui não inclui credenciais nem resultados de células.

## Próximas etapas

- Concluir o treinamento completo no Colab.
- Avaliar o `best.pt` no conjunto de teste reservado.
- Registrar métricas e examinar a matriz de confusão por classe.
- Verificar predições com fotos novas, fora do dataset.
- Integrar o modelo à aplicação e tratar entradas fora do escopo.
