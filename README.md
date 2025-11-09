[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

# Pipeline TROG-2 Eye-Tracking Analysis

Pipeline completo de análise de dados de eye-tracking para avaliação de compreensão gramatical usando o teste TROG-2 (Test for Reception of Grammar). Desenvolvido para comparação entre grupos controle e afásico.

## 📋 Índice

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura de Dados](#estrutura-de-dados)
- [Como Executar](#como-executar)
- [Processos e Análises](#processos-e-análises)
- [Métricas Calculadas](#métricas-calculadas)
- [Estrutura de Saída](#estrutura-de-saída)
- [Interpretação dos Resultados](#interpretação-dos-resultados)

---

## 🔧 Requisitos

### Dependências Python

```bash
numpy>=1.23,<3.0
pandas>=2.1,<3.0
matplotlib>=3.8,<4.0
scipy>=1.11,<2.0  # Opcional, mas recomendado para testes estatísticos
```

### Requisitos de Sistema

- Python 3.8 ou superior
- Sistema operacional: Windows, Linux ou macOS
- Memória RAM: 4GB mínimo (8GB recomendado para datasets grandes)

---

## 📦 Instalação

### 1. Clone ou baixe o script

```bash
# Se usando git
git clone <seu-repositorio>
cd <diretorio-do-projeto>

# Ou simplesmente baixe o arquivo pipeline_trog2_final.py
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**Ou instale manualmente:**

```bash
pip install numpy pandas matplotlib scipy
```

### 4. Verifique a instalação

```bash
python -c "import numpy, pandas, matplotlib, scipy; print('Instalação OK!')"
```

---

## 📁 Estrutura de Dados

### Organização dos Arquivos

```
seu-projeto/
├── data/
│   ├── controle/           # Grupo controle
│   │   ├── paciente_01.csv
│   │   ├── paciente_02.csv
│   │   └── ...
│   └── afasico/            # Grupo afásico
│       ├── paciente_A1.csv
│       ├── paciente_A2.csv
│       └── ...
├── pipeline_trog2_final.py
├── requirements.txt
└── README.md
```

### Formato dos Arquivos CSV

Cada arquivo CSV deve conter as seguintes colunas:

| Coluna | Descrição | Tipo | Exemplo |
|--------|-----------|------|---------|
| **Key** | Tecla pressionada (0=nenhuma, 1-4=resposta) | int | 0, 1, 2, 3, 4 |
| **Timestamp** | Tempo em microssegundos | int | 1234567890 |
| **X** | Coordenada X do olhar | float | 512.5 |
| **Y** | Coordenada Y do olhar | float | 384.2 |
| **Stimuli** | **OBRIGATÓRIO**: Nome do estímulo no formato `trog*.png` | string | trog1.png, trog-2.png |

**⚠️ IMPORTANTE**: A coluna **Stimuli** DEVE seguir o padrão `trog*.png`. Linhas com outros formatos serão automaticamente removidas.

**Exemplos válidos de Stimuli:**
- ✅ `trog1.png`
- ✅ `trog-2.png`
- ✅ `trog_03.png`
- ✅ `TROG10.PNG`

**Exemplos inválidos (serão removidos):**
- ❌ `1`, `2`, `teste`
- ❌ `trog1` (sem .png)
- ❌ `desconhecido`

### Nomenclatura dos Arquivos

O nome do arquivo deve seguir o padrão: `grupo_identificador.csv`

Exemplos:
- `controle_24.csv` → Paciente ID: `24`
- `afasico_A01.csv` → Paciente ID: `A01`
- `saudavel_maria.csv` → Paciente ID: `maria`

---

## 🚀 Como Executar

### Configuração Básica

Edite as seguintes linhas no início do script `pipeline_trog2_final.py`:

```python
# Caminhos das pastas de entrada
PASTA_CONTROLE = "data/controle"  # Pasta com CSVs do grupo controle
PASTA_AFASICO  = "data/afasico"   # Pasta com CSVs do grupo afásico
SAIDA_DIR      = "resultados"     # Pasta onde os resultados serão salvos

# Parâmetros do algoritmo I-VT (Velocity-Threshold)
PERCENTIL_LIMIAR_VEL = 85         # Limiar de velocidade (percentil)
VEL_MIN = 0.5                     # Velocidade mínima (unidades normalizadas/s)
FIX_MIN_S = 0.100                 # Duração mínima de fixação (segundos)

# Configuração de dados
TIMESTAMP_E_MICROSSEGUNDOS = True # True se timestamp em μs, False se em s
NORMALIZAR_COORDS = True          # Normaliza X,Y para [0,1]
```

### Execução

```bash
# Ativa o ambiente virtual (se criado)
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Executa o pipeline
python pipeline_trog2_final.py
```

### Saída no Console

O script exibirá o progresso em tempo real:

```
============================================================
PIPELINE TROG-2 EYE-TRACKING ANALYSIS
Análises: Individual | Por Grupo | Agrupada | Comparações
============================================================

ETAPA 1: Carregamento de dados
------------------------------------------------------------
[INFO] Carregando 5 arquivos de data/controle...
  [INFO] Validação Stimuli: 12 linhas removidas (formato inválido)
         Exemplos inválidos: ['desconhecido', '1']
  [OK] controle_01.csv - 8532 amostras válidas
  ...
[OK] Total de 42660 amostras válidas carregadas do grupo 'controle'
[OK] Apenas linhas com Stimuli no formato 'trog*.png' foram mantidas

...

============================================================
PIPELINE CONCLUÍDO COM SUCESSO!
============================================================
```

---

## 🔬 Processos e Análises

### 1. **Carregamento e Limpeza de Dados**

#### a) Validação de Stimuli
- Remove linhas onde `Stimuli` não segue o padrão `trog*.png`
- Exibe relatório de linhas removidas

#### b) Conversão de Tipos
- Converte `Key`, `Timestamp`, `X`, `Y` para tipos numéricos
- Substitui valores NaN por zero
- Converte timestamp de microssegundos para segundos

#### c) Remoção de Outliers (IQR)
- Calcula quartis Q1 e Q3 para X e Y
- Remove pontos fora do intervalo: `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
- Aplicado por paciente individualmente

#### d) Normalização de Coordenadas
- Normaliza X e Y para o intervalo [0, 1]
- Fórmula: `X_norm = (X - X_min) / (X_max - X_min)`
- Aplicado por paciente individualmente

### 2. **Detecção de Sacadas e Fixações (Algoritmo I-VT)**

O algoritmo **I-VT (Identification by Velocity Threshold)** classifica cada amostra como sacada ou fixação baseado na velocidade do olhar.

#### Cálculo da Velocidade

Para cada par de pontos consecutivos:

```
Δx = X[i] - X[i-1]
Δy = Y[i] - Y[i-1]
Δt = t[i] - t[i-1]

velocidade = √(Δx² + Δy²) / Δt
```

#### Limiar Adaptativo

O limiar de velocidade é calculado dinamicamente:

```
limiar_vt = max(percentil_85(velocidades), VEL_MIN)
```

- **Percentil 85**: Pega o valor de velocidade que 85% dos dados ficam abaixo
- **VEL_MIN**: Garante um piso mínimo de 0.5 unidades/s

#### Classificação

```
SE velocidade ≤ limiar_vt:
    classificar como FIXAÇÃO
SENÃO:
    classificar como SACADA
```

#### Filtragem de Fixações

Apenas fixações com duração ≥ 100ms são consideradas válidas.

### 3. **Janela de Exploração**

Para cada estímulo, identifica:
- **t₀**: Início do estímulo (primeiro timestamp)
- **t_resp**: Momento da resposta (primeiro Key ≠ 0)
- **Janela de exploração**: Período entre t₀ e t_resp onde Key = 0

### 4. **Segmentação de Eventos**

Agrupa amostras consecutivas do mesmo tipo (fixação ou sacada) em segmentos:

| Segmento | Tipo | Duração | X_início | Y_início | X_fim | Y_fim |
|----------|------|---------|----------|----------|-------|-------|
| 1 | fix | 0.245s | 0.34 | 0.52 | 0.35 | 0.53 |
| 2 | sac | 0.032s | 0.35 | 0.53 | 0.67 | 0.41 |
| 3 | fix | 0.312s | 0.67 | 0.41 | 0.66 | 0.42 |

---

## 📊 Métricas Calculadas

### Por Estímulo (Individual)

| Métrica | Descrição | Fórmula/Método |
|---------|-----------|----------------|
| **resposta** | Tecla pressionada (1-4 ou NaN) | Primeiro Key ≠ 0 |
| **tempo_resposta_s** | Tempo até responder (segundos) | t_resp - t₀ |
| **n_sacadas** | Número de sacadas | Contagem de segmentos tipo "sac" |
| **tempo_medio_sacada_s** | Duração média das sacadas | Σ(duração_sacadas) / n_sacadas |
| **n_fixacoes** | Número de fixações válidas | Contagem de fixações ≥ 100ms |
| **duracao_media_fix_s** | Duração média das fixações | Σ(duração_fixações) / n_fixacoes |
| **dispersao_area** | Área do bounding box | (X_max - X_min) × (Y_max - Y_min) |
| **limiar_vt** | Limiar de velocidade usado | percentil_85(velocidades) |

### Agregadas (Consolidadas)

Calculadas somando/mediando todos os estímulos de um paciente:

| Métrica | Cálculo |
|---------|---------|
| **n_estimulos** | Contagem de estímulos únicos |
| **tempo_resposta_medio_s** | Média dos tempos de resposta |
| **total_sacadas** | Soma de todas as sacadas |
| **tempo_medio_sacada_s** | Média ponderada das sacadas |
| **total_fixacoes** | Soma de todas as fixações |
| **duracao_media_fix_s** | Média ponderada das fixações |
| **dispersao_media** | Média das dispersões |

### Comparações Estatísticas

#### Estatísticas Descritivas
- **Média**: `μ = (Σx) / n`
- **Desvio Padrão**: `σ = √[Σ(x - μ)² / (n-1)]`

#### Teste Mann-Whitney U (Não-Paramétrico)

Compara dois grupos independentes sem assumir distribuição normal:

**Hipóteses:**
- H₀: As distribuições dos dois grupos são iguais
- H₁: As distribuições dos dois grupos são diferentes

**Interpretação do p-valor:**
- p < 0.05: Diferença estatisticamente significativa
- p ≥ 0.05: Não há evidência de diferença significativa

---

## 📂 Estrutura de Saída

```
resultados/
├── controle/
│   ├── pacientes/                              # ANÁLISE INDIVIDUAL
│   │   ├── pac001/
│   │   │   ├── stim_1/
│   │   │   │   ├── scatter.png                 # Dispersão X×Y
│   │   │   │   ├── heatmap.png                 # Mapa de calor
│   │   │   │   ├── timeline.png                # Linha do tempo fix/sac
│   │   │   │   └── hist_fix.png                # Histograma de fixações
│   │   │   ├── stim_2/, ..., stim_32/
│   │   │   ├── consolidado/                    # Todos os testes juntos
│   │   │   │   ├── scatter_all.png
│   │   │   │   ├── heatmap_all.png
│   │   │   │   ├── timeline_all.png
│   │   │   │   └── hist_fix_all.png
│   │   │   └── pac001_metricas.csv             # Métricas por estímulo
│   │   └── pac002/, pac003/, ...
│   │
│   ├── stim_1/                                 # ANÁLISE POR GRUPO
│   │   ├── scatter_grupo.png                   # Todos os pacientes
│   │   ├── heatmap_grupo.png
│   │   ├── timeline_grupo.png
│   │   ├── hist_fix_grupo.png
│   │   └── correlacao_disp_sac.png             # Dispersão × Sacadas
│   ├── stim_2/, ..., stim_32/
│   │
│   ├── analise_agrupada/                       # ANÁLISE CONSOLIDADA
│   │   ├── scatter_consolidado.png             # Tudo junto
│   │   ├── heatmap_consolidado.png
│   │   ├── timeline_consolidado.png
│   │   ├── hist_fix_consolidado.png
│   │   ├── correlacao_consolidada.png
│   │   └── controle_metricas_agrupadas.csv     # Métricas agregadas
│   │
│   ├── controle_concat.csv                     # Dados brutos concatenados
│   └── controle_metricas_por_paciente_por_estimulo.csv
│
├── afasico/
│   └── [mesma estrutura do controle]
│
├── comparacoes_individuais/                    # COMPARAÇÕES 1×1
│   ├── comparacoes_individuais_detalhadas.csv  # Cada controle vs cada afásico
│   └── resumo_comparacoes_individuais.csv      # Estatísticas das diferenças
│
├── inter_grupo_estatisticas_por_estimulo.csv   # Controle vs Afásico (por teste)
└── analise_agrupada/
    └── inter_grupo_estatisticas_agrupadas.csv  # Controle vs Afásico (consolidado)
```

### Arquivos CSV Principais

#### 1. `[grupo]_metricas_por_paciente_por_estimulo.csv`

Métricas detalhadas de cada paciente em cada estímulo.

**Colunas:**
- paciente_id, grupo, Stimuli
- resposta, tempo_resposta_s
- n_sacadas, tempo_medio_sacada_s
- n_fixacoes, duracao_media_fix_s
- dispersao_area, limiar_vt

#### 2. `[grupo]_metricas_agrupadas.csv`

Métricas consolidadas por paciente (média de todos os testes).

**Colunas:**
- paciente_id, grupo, n_estimulos
- tempo_resposta_medio_s
- total_sacadas, tempo_medio_sacada_s
- total_fixacoes, duracao_media_fix_s
- dispersao_media

#### 3. `inter_grupo_estatisticas_por_estimulo.csv`

Comparação estatística entre grupos para cada estímulo.

**Colunas:**
- Stimuli, Metrica
- Controle_media, Controle_desvio
- Afasico_media, Afasico_desvio
- MannWhitney_U, p_value

#### 4. `comparacoes_individuais_detalhadas.csv`

Diferenças diretas entre cada par de pacientes.

**Colunas:**
- paciente_controle, paciente_afasico, Stimuli
- diff_tempo_resposta, diff_n_sacadas, diff_n_fixacoes
- diff_dur_fix, diff_dispersao
- ctrl_tempo, afa_tempo, ctrl_sacadas, afa_sacadas

### Gráficos Gerados

#### 1. **Scatter (Dispersão X×Y)**
Mostra a distribuição espacial do olhar na tela.

#### 2. **Heatmap (Mapa de Calor)**
Visualiza áreas de maior concentração do olhar (densidade).

#### 3. **Timeline (Linha do Tempo)**
Mostra alternância entre fixações (0) e sacadas (1) ao longo do tempo.

#### 4. **Histograma de Fixações**
Distribuição das durações de fixações válidas (≥100ms).

#### 5. **Correlação Dispersão × Sacadas**
Gráfico de dispersão mostrando relação entre área explorada e número de sacadas.

---

## 📈 Interpretação dos Resultados

### Análise Individual

**Para cada paciente, verifique:**

1. **Padrões visuais consistentes?**
   - Heatmaps mostram concentração em áreas específicas?
   - Scatter mostra exploração uniforme ou focada?

2. **Tempo de resposta adequado?**
   - Valores muito altos podem indicar dificuldade
   - Valores muito baixos podem indicar impulsividade

3. **Número de sacadas/fixações:**
   - Muitas sacadas = busca visual intensa
   - Poucas sacadas = processamento rápido ou desengajamento

### Comparação Entre Grupos

**Analise as estatísticas inter-grupo:**

1. **Diferenças significativas (p < 0.05)?**
   - Indica que os grupos se comportam diferentemente
   
2. **Quais métricas diferem?**
   - Tempo de resposta: eficiência no processamento
   - Número de sacadas: estratégia de busca visual
   - Duração de fixações: profundidade do processamento

3. **Direção da diferença:**
   - Controle_media > Afasico_media
   - Ou vice-versa

### Exemplo de Interpretação

```csv
Stimuli,Metrica,Controle_media,Controle_desvio,Afasico_media,Afasico_desvio,p_value
trog1.png,tempo_resposta_s,3.245,0.832,5.678,1.234,0.023
```

**Interpretação:**
- Grupo controle responde em média 3.24s (±0.83s)
- Grupo afásico responde em média 5.68s (±1.23s)
- p = 0.023 < 0.05: **diferença significativa**
- **Conclusão**: Afásicos demoram ~75% mais tempo para responder ao estímulo 1

### Comparações Individuais

Use o arquivo `comparacoes_individuais_detalhadas.csv` para:

1. **Identificar pares similares:**
   - Diferenças próximas de zero indicam desempenho similar

2. **Encontrar outliers:**
   - Pacientes afásicos com desempenho próximo ao controle
   - Pacientes controle com desempenho atípico

3. **Análise caso-a-caso:**
   - Cada linha mostra comparação direta entre 2 indivíduos

---

## ⚙️ Parâmetros Ajustáveis

### Algoritmo I-VT

```python
PERCENTIL_LIMIAR_VEL = 85  # Percentil para limiar de velocidade (70-90 típico)
VEL_MIN = 0.5              # Velocidade mínima em unid. normalizadas/s
FIX_MIN_S = 0.100          # Duração mínima de fixação em segundos (60-200ms típico)
```

**Efeitos:**
- ↑ PERCENTIL → Mais amostras classificadas como fixação
- ↓ PERCENTIL → Mais amostras classificadas como sacada
- ↑ FIX_MIN_S → Menos fixações válidas (mais rigoroso)

### Visualização

```python
HEAT_BINS = 60  # Resolução do heatmap (20-100 típico)
```

**Efeitos:**
- ↑ HEAT_BINS → Heatmap mais detalhado (mas mais ruidoso)
- ↓ HEAT_BINS → Heatmap mais suavizado (pode perder detalhes)

### Outliers

Modificar na função `limpar_e_outliers()`:

```python
# Padrão: 1.5 × IQR
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

# Mais rigoroso (remove mais outliers):
lo, hi = q1 - 1.0 * iqr, q3 + 1.0 * iqr

# Mais permissivo (remove menos outliers):
lo, hi = q1 - 2.0 * iqr, q3 + 2.0 * iqr
```

---

## 🐛 Solução de Problemas

### Erro: "Pasta não encontrada"

```
[AVISO] Pasta não encontrada: data/controle
```

**Solução:** Verifique os caminhos das pastas no início do script.

### Erro: "Nenhum arquivo CSV encontrado"

**Solução:** 
- Verifique se há arquivos .csv nas pastas
- Verifique se a extensão está em minúscula (.csv, não .CSV)

### Erro: "Coluna X sem colunas: ['Stimuli']"

**Solução:** Adicione a coluna Stimuli ao CSV com valores no formato `trog*.png`

### Aviso: "X linhas removidas (formato inválido)"

**Causa:** Linhas com Stimuli que não segue o padrão `trog*.png`

**Solução:** 
- Corrija os valores na coluna Stimuli
- Ou remova as linhas inválidas manualmente

### Erro: "ModuleNotFoundError: No module named 'scipy'"

**Solução:** 

```bash
pip install scipy
```

Ou execute sem scipy (não terá p-valores):

```python
# O script continua funcionando, apenas sem testes estatísticos
```

### Gráficos não aparecem / Erro de display

**Causa:** Ambiente sem interface gráfica (servidor)

**Solução:** O script já usa backend 'Agg' do matplotlib (salva PNG sem GUI)

---

## 📞 Suporte e Contato

Para dúvidas, sugestões ou reportar problemas:

- **Autor:** Jonas - CESCOTECH
- **Projeto:** Pipeline TROG-2 Eye-Tracking Analysis
- **Instituição:** UNIVALI
- **Ano:** 2025

---

## 📝 Citação

Se você usar este pipeline em pesquisas acadêmicas, por favor cite:

```bibtex
@software{pipeline_trog2_2025,
  author = {Jonas, CESCOTECH},
  title = {Pipeline TROG-2 Eye-Tracking Analysis},
  year = {2025},
  institution = {UNIVALI},
  type = {Software}
}
```

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

### Resumo da Licença MIT:

✅ **Permitido:**
- ✓ Uso comercial
- ✓ Modificação
- ✓ Distribuição
- ✓ Uso privado

⚠️ **Condições:**
- Incluir a licença e copyright em cópias
- Manter atribuição ao autor original

🛡️ **Limitações:**
- Sem garantia
- Sem responsabilidade do autor

---

## 🔄 Histórico de Versões

### v1.0 (2025-01-XX)
- ✅ Análise individual por paciente
- ✅ Análise por grupo (segmentada por estímulo)
- ✅ Análise agrupada (consolidada)
- ✅ Comparações estatísticas inter-grupo
- ✅ Comparações individuais (1×1)
- ✅ Validação obrigatória de Stimuli (trog*.png)
- ✅ Algoritmo I-VT para detecção de fixações/sacadas
- ✅ Suporte para 1 ou múltiplos arquivos CSV por grupo
- ✅ Geração automática de gráficos (PNG)
- ✅ Testes estatísticos (Mann-Whitney U)

---

**Desenvolvido com ❤️ para pesquisa em afasiologia e neurociências cognitivas**