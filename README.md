# Predição de Métodos de Pagamento Brasileiros

## Dataset
O dataset utilizado neste projeto é <a href="https://www.kaggle.com/datasets/clovisdalmolinvieira/brazilian-payment-methods?resource=download">"Métodos de Pagamento Brasileiros"</a> do Kaggle, que contém dados de séries temporais sobre diferentes métodos de transação no Brasil ao longo do tempo. O dataset acompanha tanto a quantidade quanto o valor monetário de vários métodos de pagamento mensalmente de 2015 a 2025.

## Descrição do Dataset
O dataset contém as seguintes colunas:
- `YearMonth`: O ano e mês do ponto de dados no formato YYYYMM
- `quantityPix`: Quantidade de transações Pix
- `valuePix`: Valor monetário das transações Pix
- `quantityTED`: Quantidade de transações TED (Transferência Eletrônica Disponível)
- `valueTED`: Valor monetário das transações TED
- `quantityTEC`: Quantidade de transações TEC
- `valueTEC`: Valor monetário das transações TEC
- `quantityBankCheck`: Quantidade de transações de cheque bancário
- `valueBankCheck`: Valor monetário das transações de cheque bancário
- `quantityBrazilianBoletoPayment`: Quantidade de transações de pagamento por Boleto brasileiro
- `valueBrazilianBoletoPayment`: Valor monetário das transações de pagamento por Boleto brasileiro
- `quantityDOC`: Quantidade de transações DOC
- `valueDOC`: Valor monetário das transações DOC

## Seleção de Métrica
Para este problema de predição de séries temporais, escolhemos o Erro Quadrático Médio (MSE) como nossa métrica de avaliação. O MSE foi selecionado porque:

1. É apropriado para problemas de regressão onde o objetivo é prever valores contínuos
2. Penaliza fortemente erros grandes, o que é importante para previsões financeiras
3. É uma métrica padrão para previsão de séries temporais que fornece feedback claro sobre o desempenho do modelo
4. As unidades do erro são comparáveis às unidades dos dados originais (após extrair a raiz quadrada)

## Implementação do Modelo
Implementamos um modelo de Rede Neural Recorrente (RNN) usando PyTorch para prever valores e quantidades futuras de transações.

### Arquitetura RNN:
- Características de entrada: Métricas de quantidade e valor para todos os métodos de pagamento (12 características)
- Camadas ocultas: 3 camadas com 128 unidades ocultas cada
- Comprimento da sequência: 24 meses de dados históricos para prever o próximo mês
- Normalização: MinMaxScaler aplicado a todas as características

### Processo de Treinamento:
- Épocas: 300
- Tamanho do lote: 32
- Otimizador: Adam com taxa de aprendizado 0.001
- Função de perda: MSE
- Divisão treino/teste: 80/20

### Resultados:
O modelo alcançou uma perda de teste (MSE) de 0.0178 após o treinamento. A perda de treinamento mostra uma diminuição constante ao longo das épocas, indicando aprendizado bem-sucedido.

Comparações visuais entre valores reais e previstos foram criadas para cada método de pagamento, mostrando a capacidade do modelo de capturar tendências tanto nas quantidades quanto nos valores das transações. As previsões para 6 meses além dos dados disponíveis demonstram a capacidade do modelo de prever o uso futuro de métodos de pagamento.