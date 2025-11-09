#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline TROG-2 (Eye-Tracking) — Controle x Afásico
- Leitura em lote por pastas (CSV)
- Data cleaning + outliers (IQR)
- Detecção sacadas/fixações (I-VT: percentil 85; fix >= 100 ms)
- Segmentação por ESTÍMULO (detecta automaticamente; até 32)
- Tabelas por PACIENTE e por GRUPO (por estímulo)
- Gráficos salvos em PNG (sem seaborn; 1 gráfico por figura)
- Estatísticas inter-grupo (média, desvio, Mann-Whitney U)

Requisitos: numpy, pandas, matplotlib, scipy (opcional: p-valor)

CORREÇÕES APLICADAS:
- NaN substituído por zero onde apropriado
- Melhorias no cálculo de velocidade I-VT
- Validações adicionais
- Tratamento robusto de casos extremos
"""

import os
import re
import glob
import warnings
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=FutureWarning)

# SciPy para o Mann-Whitney; se não houver, o script segue sem p-valor
try:
    from scipy.stats import mannwhitneyu
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================
# Parâmetros principais
# =========================
PASTA_CONTROLE = "data/controle"
PASTA_AFASICO  = "data/afasico"
SAIDA_DIR      = "resultados"          # raiz de saída (CSV + PNGs)

# I-VT
PERCENTIL_LIMIAR_VEL = 85              # limiar adaptativo (percentil da velocidade)
VEL_MIN = 0.5                           # piso mínimo (unid. norm./s)
FIX_MIN_S = 0.100                       # duração mínima de fixação (s)

# Colunas esperadas nos CSV
COLS_ESPERADAS = ["Key", "Timestamp", "X", "Y", "Stimuli"]
TIMESTAMP_E_MICROSSEGUNDOS = True       # True se Timestamp vem em microssegundos

# Heatmap
HEAT_BINS = 60                          # resolução do histogram2d

# Normalização de coordenadas
NORMALIZAR_COORDS = True                # True para normalizar X e Y para [0, 1]


# =========================
# Utilidades gerais
# =========================
def garantir_dir(path: str) -> None:
    """Cria diretório se não existir."""
    os.makedirs(path, exist_ok=True)


def extrair_paciente_id(nome_arquivo: str) -> str:
    """
    Extrai o trecho após o primeiro "_" do nome do arquivo, sem a extensão.
    Ex.: "saudavel_24_trog.csv" -> "24_trog"
    Se não houver "_", retorna o nome completo sem extensão.
    """
    base = os.path.basename(nome_arquivo)
    stem = os.path.splitext(base)[0]
    
    if "_" in stem:
        return stem.split("_", 1)[1]
    
    return stem


def validar_stimuli(df: pd.DataFrame) -> pd.DataFrame:
    """
    VALIDAÇÃO OBRIGATÓRIA: Stimuli deve seguir o padrão trog*.png
    Remove linhas que não obedecem este padrão.
    
    Padrões válidos: trog1.png, trog-2.png, trog_03.png, trog04.png, etc.
    Padrões inválidos: desconhecido, vazio, números simples, etc.
    """
    if df.empty:
        return df
    
    # Padrão: começa com "trog" (case insensitive) e termina com ".png"
    padrao = re.compile(r"^trog.*\.png$", re.IGNORECASE)
    
    antes = len(df)
    
    # Marca linhas válidas
    mask_valido = df["Stimuli"].astype(str).apply(lambda x: bool(padrao.match(x)))
    
    df_valido = df[mask_valido].copy()
    removidas = antes - len(df_valido)
    
    if removidas > 0:
        print(f"  [INFO] Validação Stimuli: {removidas} linhas removidas (formato inválido)")
        # Mostra exemplos de valores removidos (primeiros 5)
        invalidos = df[~mask_valido]["Stimuli"].unique()[:5]
        print(f"         Exemplos inválidos: {list(invalidos)}")
    
    return df_valido


def normalizar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza coordenadas X e Y para o intervalo [0, 1] por paciente.
    CORREÇÃO: Trata casos onde não há variação nas coordenadas.
    """
    if not NORMALIZAR_COORDS or df.empty:
        return df
    
    def _normalizar_grupo(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        x_min, x_max = g["X"].min(), g["X"].max()
        y_min, y_max = g["Y"].min(), g["Y"].max()
        
        # Evitar divisão por zero
        if x_max - x_min > 1e-10:
            g["X"] = (g["X"] - x_min) / (x_max - x_min)
        else:
            g["X"] = 0.5
            
        if y_max - y_min > 1e-10:
            g["Y"] = (g["Y"] - y_min) / (y_max - y_min)
        else:
            g["Y"] = 0.5
            
        return g
    
    # Verifica se há múltiplos pacientes
    if "paciente_id" in df.columns and len(df["paciente_id"].unique()) > 1:
        return df.groupby("paciente_id", group_keys=False).apply(_normalizar_grupo)
    else:
        # Aplica normalização diretamente se houver apenas 1 paciente
        return _normalizar_grupo(df)


def carregar_pasta_csv(pasta: str, grupo: str) -> pd.DataFrame:
    """
    Carrega todos os arquivos CSV de uma pasta e concatena em um único DataFrame.
    
    FUNCIONA COM:
    - 1 arquivo: Carrega o único arquivo CSV da pasta
    - N arquivos: Carrega e concatena todos os CSVs da pasta
    
    CORREÇÃO: Substituição de NaN por zero onde apropriado.
    """
    if not os.path.exists(pasta):
        print(f"[AVISO] Pasta não encontrada: {pasta}")
        return pd.DataFrame(columns=["Key","Timestamp","t_sec","X","Y","Stimuli","paciente_id","grupo"])
    
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    
    if not arquivos:
        print(f"[AVISO] Nenhum arquivo CSV encontrado em: {pasta}")
        return pd.DataFrame(columns=["Key","Timestamp","t_sec","X","Y","Stimuli","paciente_id","grupo"])
    
    if len(arquivos) == 1:
        print(f"[INFO] Carregando 1 arquivo de {pasta}...")
    else:
        print(f"[INFO] Carregando {len(arquivos)} arquivos de {pasta}...")
    dfs = []
    
    for arq in arquivos:
        try:
            # Tenta ler com vírgula, depois ponto-e-vírgula
            try:
                df = pd.read_csv(arq)
            except Exception:
                df = pd.read_csv(arq, sep=";")

            # Normaliza cabeçalho (remove espaços)
            df.columns = [c.strip() for c in df.columns]
            
            # Mapeia colunas case-insensitive
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == "key":
                    col_map[col] = "Key"
                elif col_lower == "timestamp":
                    col_map[col] = "Timestamp"
                elif col_lower == "x":
                    col_map[col] = "X"
                elif col_lower == "y":
                    col_map[col] = "Y"
                elif col_lower in ["stimuli", "stimulus", "estímulo"]:
                    col_map[col] = "Stimuli"
            
            df = df.rename(columns=col_map)

            # Checagem de colunas obrigatórias
            faltando = [c for c in COLS_ESPERADAS if c not in df.columns]
            if faltando:
                print(f"[ERRO] {arq} sem colunas: {faltando}. Pulando arquivo.")
                continue

            # CORREÇÃO: Conversão de tipos com substituição de NaN por zero
            df["Key"] = pd.to_numeric(df["Key"], errors="coerce").fillna(0).astype(int)
            df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0)
            df["X"] = pd.to_numeric(df["X"], errors="coerce").fillna(0)
            df["Y"] = pd.to_numeric(df["Y"], errors="coerce").fillna(0)

            # Timestamp para segundos
            if TIMESTAMP_E_MICROSSEGUNDOS:
                df["t_sec"] = df["Timestamp"] / 1e6
            else:
                df["t_sec"] = df["Timestamp"].astype(float)

            # Padronizar Stimuli como string (remover espaços)
            df["Stimuli"] = df["Stimuli"].astype(str).str.strip()
            
            # VALIDAÇÃO OBRIGATÓRIA: Remover linhas onde Stimuli não segue padrão trog*.png
            df = validar_stimuli(df)
            
            if df.empty:
                print(f"  [AVISO] {arq} não possui linhas com Stimuli válido (trog*.png). Pulando arquivo.")
                continue

            # Adicionar identificadores
            df["paciente_id"] = extrair_paciente_id(arq)
            df["grupo"] = grupo

            # Selecionar colunas relevantes
            dfs.append(df[["Key", "Timestamp", "t_sec", "X", "Y", "Stimuli", "paciente_id", "grupo"]])
            
            print(f"  [OK] {os.path.basename(arq)} - {len(df)} amostras válidas")

        except Exception as e:
            print(f"[ERRO] Falha ao processar {arq}: {str(e)}")
            continue

    if not dfs:
        print(f"[AVISO] Nenhum arquivo válido carregado de {pasta}")
        return pd.DataFrame(columns=["Key","Timestamp","t_sec","X","Y","Stimuli","paciente_id","grupo"])
    
    resultado = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Total de {len(resultado)} amostras válidas carregadas do grupo '{grupo}'")
    print(f"[OK] Apenas linhas com Stimuli no formato 'trog*.png' foram mantidas\n")
    return resultado


def limpar_e_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dados inválidos e outliers usando IQR.
    CORREÇÃO: Mantém dados válidos (zeros são mantidos), remove apenas outliers extremos.
    CORREÇÃO: Funciona com 1 ou mais pacientes.
    """
    if df.empty:
        return df
    
    # Remove apenas linhas onde TODAS as coordenadas são zero (provavelmente erro)
    df = df[~((df["X"] == 0) & (df["Y"] == 0))].copy()
    
    # Ordena
    df = df.sort_values(["paciente_id", "t_sec"]).reset_index(drop=True)
    
    print(f"[INFO] Limpeza de dados: {len(df)} amostras após remover dados inválidos")

    # Remove outliers por paciente via IQR (X e Y)
    def _iqr_mask(g: pd.DataFrame, col: str) -> pd.Series:
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (g[col] >= lo) & (g[col] <= hi)

    # Verifica se há múltiplos pacientes
    if "paciente_id" in df.columns and len(df["paciente_id"].unique()) > 1:
        mask_x = df.groupby("paciente_id", group_keys=False).apply(
            lambda g: _iqr_mask(g, "X"),
            include_groups=False
        )
        mask_y = df.groupby("paciente_id", group_keys=False).apply(
            lambda g: _iqr_mask(g, "Y"),
            include_groups=False
        )
    else:
        # Aplica IQR diretamente se houver apenas 1 paciente
        mask_x = _iqr_mask(df, "X")
        mask_y = _iqr_mask(df, "Y")
    
    antes = len(df)
    df = df[mask_x & mask_y].copy()
    depois = len(df)
    print(f"[INFO] Remoção de outliers (IQR): {antes - depois} amostras removidas ({depois} restantes)")

    # Elimina amostras com dt==0 (duplicatas de tempo)
    if "paciente_id" in df.columns and len(df["paciente_id"].unique()) > 1:
        df["dt"] = df.groupby("paciente_id")["t_sec"].diff()
    else:
        df["dt"] = df["t_sec"].diff()
    
    # Remove pontos onde dt é exatamente 0
    antes = len(df)
    df = df[(df["dt"] != 0) | (df["dt"].isna())].copy()
    depois = len(df)
    if antes - depois > 0:
        print(f"[INFO] Remoção de timestamps duplicados: {antes - depois} amostras removidas")
    
    df = df.drop(columns=["dt"])
    
    return df


def janela_exploracao(df_pac: pd.DataFrame) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Retorna:
      - explore: janela (Key==0) até o 1º Key!=0
      - t0: tempo inicial
      - t_resp: tempo da resposta (ou fim)
      - tempo_resposta: t_resp - t0
      - resposta: valor Key (1..4) ou NaN
    
    CORREÇÃO: Tratamento robusto de casos extremos.
    """
    if df_pac.empty:
        return df_pac, np.nan, np.nan, np.nan, np.nan
    
    df_pac = df_pac.sort_values("t_sec").reset_index(drop=True)
    t0 = float(df_pac["t_sec"].iloc[0])
    
    # Procura primeira resposta (Key != 0)
    respostas = df_pac[df_pac["Key"] != 0]
    
    if not respostas.empty:
        idx = respostas.index[0]
        t_resp = float(df_pac.loc[idx, "t_sec"])
        resposta = int(df_pac.loc[idx, "Key"])
        # Janela de exploração: até a resposta (exclusive)
        explore = df_pac[(df_pac["Key"] == 0) & (df_pac.index < idx)].copy()
    else:
        # Sem resposta: toda a janela é exploração
        t_resp = float(df_pac["t_sec"].iloc[-1])
        resposta = np.nan
        explore = df_pac[df_pac["Key"] == 0].copy()
    
    tempo_resposta = t_resp - t0
    
    return explore, t0, t_resp, tempo_resposta, resposta


def rotular_ivt(explore: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    I-VT com limiar no percentil definido + piso VEL_MIN.
    Retorna DataFrame com labels e o limiar de velocidade usado.
    
    CORREÇÃO: Cálculo de velocidade mais robusto.
    """
    if explore.empty or len(explore) < 3:
        explore = explore.copy()
        explore["dx"] = 0.0
        explore["dy"] = 0.0
        explore["dt"] = 0.0
        explore["speed"] = 0.0
        explore["label"] = "fix"
        explore["seg_id"] = 1
        return explore, VEL_MIN

    explore = explore.copy().sort_values("t_sec").reset_index(drop=True)

    # Calcula diferenças e velocidade
    dx = explore["X"].diff().fillna(0.0)
    dy = explore["Y"].diff().fillna(0.0)
    dt = explore["t_sec"].diff()
    
    # CORREÇÃO: Tratamento robusto de dt
    # Substitui dt=0 e dt muito pequeno por um valor mínimo
    dt = dt.fillna(0.0)
    dt = dt.replace(0, 1e-6)
    dt = dt.apply(lambda x: max(x, 1e-6))
    
    # Calcula velocidade
    speed = np.sqrt(dx**2 + dy**2) / dt
    
    # CORREÇÃO: Trata valores infinitos ou muito grandes
    speed = speed.replace([np.inf, -np.inf], 0.0)
    speed = speed.fillna(0.0)

    explore["dx"] = dx.values
    explore["dy"] = dy.values
    explore["dt"] = dt.values
    explore["speed"] = speed.values
    
    # Limiar adaptativo
    if explore["speed"].max() > 0:
        vt = np.nanpercentile(explore["speed"].values, PERCENTIL_LIMIAR_VEL)
        vt = max(vt, VEL_MIN)
    else:
        vt = VEL_MIN

    # Classificação
    explore["label"] = np.where(explore["speed"] <= vt, "fix", "sac")
    
    # Segmentação
    explore["seg_id"] = (explore["label"] != explore["label"].shift(1)).cumsum()
    
    return explore, vt


def segmentos_por_label(explore_ivt: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa segmentos contínuos de fixação/sacada.
    """
    if explore_ivt.empty:
        return pd.DataFrame(columns=["seg_id", "label", "t_start", "t_end", "n", 
                                     "x_start", "y_start", "x_end", "y_end", "dur"])
    
    seg = (
        explore_ivt.groupby(["seg_id", "label"])
        .agg(
            t_start=("t_sec", "min"),
            t_end=("t_sec", "max"),
            n=("t_sec", "size"),
            x_start=("X", "first"),
            y_start=("Y", "first"),
            x_end=("X", "last"),
            y_end=("Y", "last"),
        ).reset_index()
    )
    seg["dur"] = seg["t_end"] - seg["t_start"]
    return seg


def dispersao_area_bb(explore: pd.DataFrame) -> float:
    """
    Calcula área do bounding box (dispersão espacial).
    """
    if explore.empty or len(explore) < 2:
        return 0.0
    
    x_range = explore["X"].max() - explore["X"].min()
    y_range = explore["Y"].max() - explore["Y"].min()
    
    return float(x_range * y_range)


# =========================
# Gráficos (salvar PNG)
# =========================
def save_scatter(df: pd.DataFrame, caminho_png: str, titulo: str):
    """Gráfico de dispersão X×Y."""
    if df.empty:
        return
    
    plt.figure(figsize=(6,6))
    plt.scatter(df["X"], df["Y"], s=4, alpha=0.4, c='steelblue')
    plt.gca().invert_yaxis()
    plt.title(titulo, fontsize=10)
    plt.xlabel("X (normalizado)")
    plt.ylabel("Y (normalizado)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=200, bbox_inches='tight')
    plt.close()


def save_heatmap(df: pd.DataFrame, caminho_png: str, titulo: str, bins: int = HEAT_BINS):
    """Heatmap de densidade visual."""
    if df.empty or len(df) < 2:
        return
    
    try:
        H, xedges, yedges = np.histogram2d(df["X"], df["Y"], bins=bins)
        H = H.T
        
        plt.figure(figsize=(6,6))
        plt.imshow(H, origin="lower",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect="auto", cmap='hot')
        plt.gca().invert_yaxis()
        plt.title(titulo, fontsize=10)
        plt.xlabel("X (normalizado)")
        plt.ylabel("Y (normalizado)")
        plt.colorbar(label="Densidade")
        plt.tight_layout()
        plt.savefig(caminho_png, dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [AVISO] Erro ao criar heatmap: {e}")


def save_correlacao_disp_sac(mets_df: pd.DataFrame, caminho_png: str, titulo: str):
    """Correlação dispersão × nº sacadas."""
    if mets_df.empty or "dispersao_area" not in mets_df.columns or "n_sacadas" not in mets_df.columns:
        return
    
    # Remove valores inválidos
    mets_valid = mets_df.dropna(subset=["dispersao_area", "n_sacadas"])
    mets_valid = mets_valid[(mets_valid["dispersao_area"] > 0) | (mets_valid["n_sacadas"] > 0)]
    
    if mets_valid.empty:
        return
    
    plt.figure(figsize=(7,5))
    x = mets_valid["dispersao_area"].values
    y = mets_valid["n_sacadas"].values
    plt.scatter(x, y, s=80, alpha=0.6, c='darkgreen')
    
    # Anotações
    for _, row in mets_valid.iterrows():
        plt.text(row["dispersao_area"]*1.01, row["n_sacadas"]*1.01, 
                 str(row["paciente_id"]), fontsize=8)
    
    plt.title(titulo, fontsize=10)
    plt.xlabel("Dispersão (área bounding box)")
    plt.ylabel("Número de sacadas")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=200, bbox_inches='tight')
    plt.close()


def save_timeline_fix_sac(grupo_df: pd.DataFrame, caminho_png: str, titulo: str):
    """Linha do tempo: 0=Fixação, 1=Sacada."""
    if grupo_df.empty:
        return
    
    linhas = []
    for _, g in grupo_df.groupby("paciente_id"):
        explore, *_ = janela_exploracao(g)
        if explore.empty:
            continue
        ivt, _ = rotular_ivt(explore)
        if not ivt.empty:
            y = (ivt["label"] == "sac").astype(int)
            linhas.append(pd.DataFrame({"t_sec": ivt["t_sec"].values, "cat": y.values}))
    
    if not linhas:
        return
    
    TL = pd.concat(linhas, ignore_index=True).sort_values("t_sec")
    
    plt.figure(figsize=(10,2.8))
    plt.plot(TL["t_sec"], TL["cat"], drawstyle="steps-pre", linewidth=0.8)
    plt.yticks([0,1], ["Fixação","Sacada"])
    plt.title(titulo, fontsize=10)
    plt.xlabel("Tempo (s)")
    plt.ylabel("")
    plt.ylim(-0.2, 1.2)
    plt.grid(True, axis='x', linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=200, bbox_inches='tight')
    plt.close()


def save_hist_fix(grupo_df: pd.DataFrame, caminho_png: str, titulo: str, bins: int = 24):
    """Distribuição das durações de fixação (>= 100ms)."""
    if grupo_df.empty:
        return
    
    all_fix = []
    for _, g in grupo_df.groupby("paciente_id"):
        explore, *_ = janela_exploracao(g)
        if explore.empty:
            continue
        ivt, _ = rotular_ivt(explore)
        seg = segmentos_por_label(ivt)
        fix = seg[(seg["label"] == "fix") & (seg["dur"] >= FIX_MIN_S)]
        if not fix.empty:
            all_fix.append(fix["dur"].values)
    
    if not all_fix:
        return
    
    arr = np.concatenate(all_fix, axis=0)
    
    plt.figure(figsize=(8,4))
    plt.hist(arr, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(titulo, fontsize=10)
    plt.xlabel("Duração (s)")
    plt.ylabel("Frequência")
    plt.grid(True, axis='y', linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho_png, dpi=200, bbox_inches='tight')
    plt.close()


# =========================
# Métricas
# =========================
def metricas_paciente_por_estimulo(df_pac_est: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de um paciente para um estímulo específico.
    """
    if df_pac_est.empty:
        return dict(
            resposta=np.nan,
            tempo_resposta_s=np.nan,
            n_sacadas=0,
            tempo_medio_sacada_s=np.nan,
            n_fixacoes=0,
            duracao_media_fix_s=np.nan,
            dispersao_area=0.0,
            limiar_vt=np.nan
        )
    
    explore, t0, t_resp, t_total, resposta = janela_exploracao(df_pac_est)
    
    if explore.empty:
        return dict(
            resposta=resposta,
            tempo_resposta_s=round(t_total, 3) if not np.isnan(t_total) else np.nan,
            n_sacadas=0,
            tempo_medio_sacada_s=np.nan,
            n_fixacoes=0,
            duracao_media_fix_s=np.nan,
            dispersao_area=0.0,
            limiar_vt=np.nan
        )
    
    explore_ivt, vt = rotular_ivt(explore)
    seg = segmentos_por_label(explore_ivt)
    fix = seg[(seg["label"] == "fix") & (seg["dur"] >= FIX_MIN_S)]
    sac = seg[seg["label"] == "sac"]
    disp = dispersao_area_bb(explore)

    return dict(
        resposta=resposta,
        tempo_resposta_s=round(t_total, 3) if not np.isnan(t_total) else np.nan,
        n_sacadas=int(len(sac)),
        tempo_medio_sacada_s=round(float(sac["dur"].mean()), 4) if len(sac) else np.nan,
        n_fixacoes=int(len(fix)),
        duracao_media_fix_s=round(float(fix["dur"].mean()), 4) if len(fix) else np.nan,
        dispersao_area=round(disp, 6),
        limiar_vt=round(vt, 4)
    )


# =========================
# Análise INDIVIDUAL por paciente
# =========================
def processar_paciente_individual(df_pac: pd.DataFrame, paciente_id: str, grupo: str) -> pd.DataFrame:
    """
    Processa um paciente individual gerando:
      - Gráficos por estímulo
      - Gráficos consolidados (todos os estímulos)
      - Métricas detalhadas
    """
    if df_pac.empty:
        print(f"[AVISO] Paciente '{paciente_id}' está vazio.")
        return pd.DataFrame()
    
    # Saída para este paciente
    out_pac = os.path.join(SAIDA_DIR, grupo, "pacientes", paciente_id)
    garantir_dir(out_pac)
    
    print(f"  → Paciente: {paciente_id}")
    
    # Normalizar coordenadas do paciente
    df_pac_norm = normalizar_coordenadas(df_pac)
    
    # Métricas por estímulo
    mets_rows = []
    estimulos = sorted(df_pac_norm["Stimuli"].unique())
    
    for stim in estimulos:
        df_stim = df_pac_norm[df_pac_norm["Stimuli"] == stim]
        stim_str = str(stim)
        
        # Pasta para este estímulo
        out_stim = os.path.join(out_pac, f"stim_{stim_str}")
        garantir_dir(out_stim)
        
        # Gráficos individuais por estímulo
        save_scatter(df_stim, os.path.join(out_stim, f"scatter.png"),
                     f"Dispersão — {paciente_id} — Estímulo {stim_str}")
        save_heatmap(df_stim, os.path.join(out_stim, f"heatmap.png"),
                     f"Heatmap — {paciente_id} — Estímulo {stim_str}")
        save_timeline_fix_sac(df_stim, os.path.join(out_stim, f"timeline.png"),
                              f"Timeline — {paciente_id} — Estímulo {stim_str}")
        save_hist_fix(df_stim, os.path.join(out_stim, f"hist_fix.png"),
                      f"Fixações — {paciente_id} — Estímulo {stim_str}")
        
        # Métricas
        m = metricas_paciente_por_estimulo(df_stim)
        m["paciente_id"] = paciente_id
        m["grupo"] = grupo
        m["Stimuli"] = stim
        mets_rows.append(m)
    
    # Gráficos CONSOLIDADOS do paciente (todos os estímulos)
    out_consol = os.path.join(out_pac, "consolidado")
    garantir_dir(out_consol)
    
    save_scatter(df_pac_norm, os.path.join(out_consol, f"scatter_all.png"),
                 f"Dispersão — {paciente_id} — Todos os Estímulos")
    save_heatmap(df_pac_norm, os.path.join(out_consol, f"heatmap_all.png"),
                 f"Heatmap — {paciente_id} — Todos os Estímulos")
    save_timeline_fix_sac(df_pac_norm, os.path.join(out_consol, f"timeline_all.png"),
                          f"Timeline — {paciente_id} — Todos os Estímulos")
    save_hist_fix(df_pac_norm, os.path.join(out_consol, f"hist_fix_all.png"),
                  f"Fixações — {paciente_id} — Todos os Estímulos")
    
    # CSV com métricas do paciente
    mets_pac = pd.DataFrame(mets_rows) if mets_rows else pd.DataFrame()
    if not mets_pac.empty:
        csv_path = os.path.join(out_pac, f"{paciente_id}_metricas.csv")
        mets_pac.to_csv(csv_path, index=False, encoding="utf-8")
    
    return mets_pac


# =========================
# Processo por GRUPO + ESTÍMULO
# =========================
def processar_grupo_segmentado(df_grupo: pd.DataFrame, grupo: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processa um grupo completo, gerando:
      - CSV concatenado
      - Métricas por paciente+estímulo
      - Gráficos por estímulo
      - Análise individual de cada paciente
    """
    if df_grupo.empty:
        print(f"[AVISO] Grupo '{grupo}' está vazio. Pulando processamento.")
        return df_grupo, pd.DataFrame()

    # Saídas
    out_root = os.path.join(SAIDA_DIR, grupo)
    garantir_dir(out_root)
    
    print(f"\n{'='*60}")
    print(f"Processando grupo: {grupo.upper()}")
    print(f"{'='*60}")

    # Normalizar coordenadas
    df_grupo = normalizar_coordenadas(df_grupo)

    # CSV concatenado do grupo
    csv_concat = os.path.join(out_root, f"{grupo}_concat.csv")
    df_grupo.to_csv(csv_concat, index=False, encoding="utf-8")
    print(f"[OK] CSV concatenado salvo: {csv_concat}")

    # ========================================
    # ANÁLISE INDIVIDUAL DE CADA PACIENTE
    # ========================================
    print(f"\n[INFO] Processando pacientes individuais...")
    pacientes = sorted(df_grupo["paciente_id"].unique())
    mets_individuais = []
    
    for pid in pacientes:
        df_pac = df_grupo[df_grupo["paciente_id"] == pid]
        mets_pac = processar_paciente_individual(df_pac, pid, grupo)
        if not mets_pac.empty:
            mets_individuais.append(mets_pac)
    
    # ========================================
    # ANÁLISE POR ESTÍMULO (GRUPO)
    # ========================================
    print(f"\n[INFO] Processando análise por estímulo (grupo)...")
    mets_rows = []
    estimulos = sorted(df_grupo["Stimuli"].unique())
    
    print(f"[INFO] Processando {len(estimulos)} estímulo(s): {estimulos}")
    
    for stim in estimulos:
        df_est = df_grupo[df_grupo["Stimuli"] == stim]
        stim_str = str(stim)
        out_stim = os.path.join(out_root, f"stim_{stim_str}")
        garantir_dir(out_stim)
        
        print(f"  → Estímulo {stim_str}: {len(df_est)} amostras")

        # Gráficos de grupo (por estímulo)
        save_scatter(df_est, os.path.join(out_stim, f"scatter_grupo.png"),
                     f"Dispersão X×Y — {grupo.title()} — Estímulo {stim_str}")
        save_heatmap(df_est, os.path.join(out_stim, f"heatmap_grupo.png"),
                     f"Heatmap — {grupo.title()} — Estímulo {stim_str}")
        save_timeline_fix_sac(df_est, os.path.join(out_stim, f"timeline_grupo.png"),
                              f"Timeline Fixação/Sacada — {grupo.title()} — Estímulo {stim_str}")
        save_hist_fix(df_est, os.path.join(out_stim, f"hist_fix_grupo.png"),
                      f"Distribuição Fixações (≥100ms) — {grupo.title()} — Estímulo {stim_str}")

        # Métricas por PACIENTE neste estímulo
        for pid in df_est["paciente_id"].unique():
            df_pac = df_est[df_est["paciente_id"] == pid]
            m = metricas_paciente_por_estimulo(df_pac)
            m["paciente_id"] = pid
            m["grupo"] = grupo
            m["Stimuli"] = stim
            mets_rows.append(m)

        # Correlação (dispersão × nº sacadas) — por estímulo
        mets_est = pd.DataFrame([r for r in mets_rows if r.get("Stimuli") == stim and r.get("grupo") == grupo])
        if not mets_est.empty and {"dispersao_area","n_sacadas"}.issubset(mets_est.columns):
            save_correlacao_disp_sac(
                mets_est,
                os.path.join(out_stim, f"correlacao_disp_sac.png"),
                f"Dispersão × Sacadas — {grupo.title()} — Estímulo {stim_str}"
            )

    # Consolidação de métricas
    if not mets_rows:
        mets_all = pd.DataFrame(
            columns=[
                "paciente_id", "grupo", "Stimuli", "resposta", "tempo_resposta_s", 
                "n_sacadas", "tempo_medio_sacada_s", "n_fixacoes", "duracao_media_fix_s", 
                "dispersao_area", "limiar_vt"
            ]
        )
    else:
        mets_all = pd.DataFrame(mets_rows)
        mets_all["Stimuli"] = mets_all["Stimuli"].astype(str)
        mets_all = mets_all.sort_values(["Stimuli", "paciente_id"]).reset_index(drop=True)

    csv_metricas = os.path.join(out_root, f"{grupo}_metricas_por_paciente_por_estimulo.csv")
    mets_all.to_csv(csv_metricas, index=False, encoding="utf-8")
    print(f"[OK] Métricas salvas: {csv_metricas}\n")
    
    return df_grupo, mets_all


def processar_grupo_agrupado(df_grupo: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """
    Processa grupo COMPLETO (todos os estímulos agregados).
    Gera análise consolidada SEM segmentação por estímulo.
    """
    if df_grupo.empty:
        print(f"[AVISO] Grupo '{grupo}' está vazio para análise agrupada.")
        return pd.DataFrame()

    out_root = os.path.join(SAIDA_DIR, grupo, "analise_agrupada")
    garantir_dir(out_root)
    
    print(f"\n{'='*60}")
    print(f"Análise AGRUPADA (consolidada): {grupo.upper()}")
    print(f"{'='*60}")

    # Gráficos do grupo COMPLETO
    print(f"[INFO] Gerando gráficos consolidados...")
    save_scatter(df_grupo, os.path.join(out_root, f"scatter_consolidado.png"),
                 f"Dispersão X×Y — {grupo.title()} — Todos os Estímulos")
    save_heatmap(df_grupo, os.path.join(out_root, f"heatmap_consolidado.png"),
                 f"Heatmap — {grupo.title()} — Todos os Estímulos")
    save_timeline_fix_sac(df_grupo, os.path.join(out_root, f"timeline_consolidado.png"),
                          f"Timeline Fixação/Sacada — {grupo.title()} — Todos os Estímulos")
    save_hist_fix(df_grupo, os.path.join(out_root, f"hist_fix_consolidado.png"),
                  f"Distribuição Fixações (≥100ms) — {grupo.title()} — Todos os Estímulos")

    # Métricas AGREGADAS por paciente (soma de todos os estímulos)
    mets_rows = []
    for pid in df_grupo["paciente_id"].unique():
        df_pac = df_grupo[df_grupo["paciente_id"] == pid]
        
        # Calcula métricas agregadas
        total_resp = 0
        total_tempo = 0.0
        total_sac = 0
        total_fix = 0
        soma_dur_sac = 0.0
        soma_dur_fix = 0.0
        soma_disp = 0.0
        n_estimulos = 0
        
        for stim in df_pac["Stimuli"].unique():
            df_stim = df_pac[df_pac["Stimuli"] == stim]
            m = metricas_paciente_por_estimulo(df_stim)
            
            if not np.isnan(m["tempo_resposta_s"]):
                total_tempo += m["tempo_resposta_s"]
                n_estimulos += 1
            
            total_sac += m["n_sacadas"]
            total_fix += m["n_fixacoes"]
            
            if not np.isnan(m["tempo_medio_sacada_s"]):
                soma_dur_sac += m["tempo_medio_sacada_s"] * m["n_sacadas"]
            
            if not np.isnan(m["duracao_media_fix_s"]):
                soma_dur_fix += m["duracao_media_fix_s"] * m["n_fixacoes"]
            
            soma_disp += m["dispersao_area"]
        
        # Médias
        tempo_medio = total_tempo / n_estimulos if n_estimulos > 0 else np.nan
        tempo_medio_sac = soma_dur_sac / total_sac if total_sac > 0 else np.nan
        dur_media_fix = soma_dur_fix / total_fix if total_fix > 0 else np.nan
        disp_media = soma_disp / n_estimulos if n_estimulos > 0 else 0.0
        
        mets_rows.append({
            "paciente_id": pid,
            "grupo": grupo,
            "n_estimulos": n_estimulos,
            "tempo_resposta_medio_s": round(tempo_medio, 3) if not np.isnan(tempo_medio) else np.nan,
            "total_sacadas": int(total_sac),
            "tempo_medio_sacada_s": round(tempo_medio_sac, 4) if not np.isnan(tempo_medio_sac) else np.nan,
            "total_fixacoes": int(total_fix),
            "duracao_media_fix_s": round(dur_media_fix, 4) if not np.isnan(dur_media_fix) else np.nan,
            "dispersao_media": round(disp_media, 6),
        })
    
    if not mets_rows:
        mets_agrup = pd.DataFrame(
            columns=[
                "paciente_id", "grupo", "n_estimulos", "tempo_resposta_medio_s",
                "total_sacadas", "tempo_medio_sacada_s", "total_fixacoes",
                "duracao_media_fix_s", "dispersao_media"
            ]
        )
    else:
        mets_agrup = pd.DataFrame(mets_rows).sort_values("paciente_id").reset_index(drop=True)
    
    # Salvar CSV
    csv_path = os.path.join(out_root, f"{grupo}_metricas_agrupadas.csv")
    mets_agrup.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Métricas agrupadas salvas: {csv_path}")
    
    # Gráfico de correlação consolidado
    if not mets_agrup.empty and {"dispersao_media","total_sacadas"}.issubset(mets_agrup.columns):
        save_correlacao_disp_sac(
            mets_agrup.rename(columns={"dispersao_media": "dispersao_area", "total_sacadas": "n_sacadas"}),
            os.path.join(out_root, f"correlacao_consolidada.png"),
            f"Dispersão × Sacadas — {grupo.title()} — Consolidado"
        )
    
    print(f"[OK] Análise agrupada concluída para {grupo}\n")
    
    return mets_agrup


# =========================
# Estatísticas inter-grupo
# =========================
METRICAS_COMPARAR = [
    "tempo_resposta_s",
    "n_sacadas",
    "tempo_medio_sacada_s",
    "n_fixacoes",
    "duracao_media_fix_s",
    "dispersao_area",
]

def estatisticas_inter_grupo(mets_ctrl: pd.DataFrame, mets_afa: pd.DataFrame) -> pd.DataFrame:
    """
    Gera estatísticas comparativas entre grupos por estímulo e métrica.
    Calcula média, desvio padrão e Mann-Whitney U test.
    """
    linhas = []
    
    if mets_ctrl.empty and mets_afa.empty:
        out = pd.DataFrame(columns=[
            "Stimuli","Metrica","Controle_media","Controle_desvio",
            "Afasico_media","Afasico_desvio","MannWhitney_U","p_value"
        ])
        garantir_dir(SAIDA_DIR)
        csv_path = os.path.join(SAIDA_DIR, "inter_grupo_estatisticas_por_estimulo.csv")
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return out

    # Todos os estímulos presentes em qualquer grupo
    stim_ctrl = set(mets_ctrl["Stimuli"].unique()) if not mets_ctrl.empty else set()
    stim_afa = set(mets_afa["Stimuli"].unique()) if not mets_afa.empty else set()
    stim_todos = sorted(stim_ctrl.union(stim_afa))
    
    print(f"\n{'='*60}")
    print("Calculando estatísticas inter-grupo")
    print(f"{'='*60}")
    print(f"Estímulos a comparar: {stim_todos}\n")
    
    for stim in stim_todos:
        ctrl = mets_ctrl[mets_ctrl["Stimuli"] == stim] if not mets_ctrl.empty else pd.DataFrame()
        afa  = mets_afa[mets_afa["Stimuli"] == stim] if not mets_afa.empty else pd.DataFrame()
        
        for m in METRICAS_COMPARAR:
            arr_c = ctrl[m].dropna().values if not ctrl.empty and m in ctrl.columns else np.array([])
            arr_a = afa[m].dropna().values if not afa.empty and m in afa.columns else np.array([])
            
            if len(arr_c) == 0 and len(arr_a) == 0:
                continue
            
            # Estatísticas descritivas
            media_c = float(np.mean(arr_c)) if len(arr_c) > 0 else np.nan
            desvp_c = float(np.std(arr_c, ddof=1)) if len(arr_c) > 1 else np.nan
            media_a = float(np.mean(arr_a)) if len(arr_a) > 0 else np.nan
            desvp_a = float(np.std(arr_a, ddof=1)) if len(arr_a) > 1 else np.nan

            # Mann-Whitney U test
            if SCIPY_OK and len(arr_c) > 0 and len(arr_a) > 0:
                try:
                    u, p = mannwhitneyu(arr_c, arr_a, alternative="two-sided")
                    uval, pval = float(u), float(p)
                except Exception as e:
                    print(f"  [AVISO] Mann-Whitney falhou para {m} (estímulo {stim}): {e}")
                    uval, pval = np.nan, np.nan
            else:
                uval, pval = np.nan, np.nan

            linhas.append({
                "Stimuli": stim,
                "Metrica": m,
                "Controle_media": round(media_c, 6) if not np.isnan(media_c) else None,
                "Controle_desvio": round(desvp_c, 6) if not np.isnan(desvp_c) else None,
                "Afasico_media": round(media_a, 6) if not np.isnan(media_a) else None,
                "Afasico_desvio": round(desvp_a, 6) if not np.isnan(desvp_a) else None,
                "MannWhitney_U": round(uval, 6) if not np.isnan(uval) else None,
                "p_value": round(pval, 6) if not np.isnan(pval) else None,
            })
    
    if not linhas:
        print("[AVISO] Nenhuma métrica válida para comparação entre grupos.")
        out = pd.DataFrame(columns=[
            "Stimuli","Metrica","Controle_media","Controle_desvio",
            "Afasico_media","Afasico_desvio","MannWhitney_U","p_value"
        ])
    else:
        out = pd.DataFrame(linhas).sort_values(["Stimuli","Metrica"]).reset_index(drop=True)
    
    garantir_dir(SAIDA_DIR)
    csv_path = os.path.join(SAIDA_DIR, "inter_grupo_estatisticas_por_estimulo.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Estatísticas inter-grupo salvas: {csv_path}\n")
    
    return out


def estatisticas_inter_grupo_agrupado(mets_ctrl_agrup: pd.DataFrame, mets_afa_agrup: pd.DataFrame) -> pd.DataFrame:
    """
    Gera estatísticas comparativas entre grupos para análise AGRUPADA (consolidada).
    Calcula média, desvio padrão e Mann-Whitney U test.
    """
    linhas = []
    
    if mets_ctrl_agrup.empty and mets_afa_agrup.empty:
        out = pd.DataFrame(columns=[
            "Metrica","Controle_media","Controle_desvio",
            "Afasico_media","Afasico_desvio","MannWhitney_U","p_value"
        ])
        garantir_dir(os.path.join(SAIDA_DIR, "analise_agrupada"))
        csv_path = os.path.join(SAIDA_DIR, "analise_agrupada", "inter_grupo_estatisticas_agrupadas.csv")
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return out
    
    print(f"\n{'='*60}")
    print("Calculando estatísticas inter-grupo AGRUPADAS")
    print(f"{'='*60}")
    
    # Métricas para comparação na análise agrupada
    metricas_agrup = [
        "tempo_resposta_medio_s",
        "total_sacadas",
        "tempo_medio_sacada_s",
        "total_fixacoes",
        "duracao_media_fix_s",
        "dispersao_media",
    ]
    
    for m in metricas_agrup:
        arr_c = mets_ctrl_agrup[m].dropna().values if not mets_ctrl_agrup.empty and m in mets_ctrl_agrup.columns else np.array([])
        arr_a = mets_afa_agrup[m].dropna().values if not mets_afa_agrup.empty and m in mets_afa_agrup.columns else np.array([])
        
        if len(arr_c) == 0 and len(arr_a) == 0:
            continue
        
        # Estatísticas descritivas
        media_c = float(np.mean(arr_c)) if len(arr_c) > 0 else np.nan
        desvp_c = float(np.std(arr_c, ddof=1)) if len(arr_c) > 1 else np.nan
        media_a = float(np.mean(arr_a)) if len(arr_a) > 0 else np.nan
        desvp_a = float(np.std(arr_a, ddof=1)) if len(arr_a) > 1 else np.nan

        # Mann-Whitney U test
        if SCIPY_OK and len(arr_c) > 0 and len(arr_a) > 0:
            try:
                u, p = mannwhitneyu(arr_c, arr_a, alternative="two-sided")
                uval, pval = float(u), float(p)
            except Exception as e:
                print(f"  [AVISO] Mann-Whitney falhou para {m}: {e}")
                uval, pval = np.nan, np.nan
        else:
            uval, pval = np.nan, np.nan

        linhas.append({
            "Metrica": m,
            "Controle_media": round(media_c, 6) if not np.isnan(media_c) else None,
            "Controle_desvio": round(desvp_c, 6) if not np.isnan(desvp_c) else None,
            "Afasico_media": round(media_a, 6) if not np.isnan(media_a) else None,
            "Afasico_desvio": round(desvp_a, 6) if not np.isnan(desvp_a) else None,
            "MannWhitney_U": round(uval, 6) if not np.isnan(uval) else None,
            "p_value": round(pval, 6) if not np.isnan(pval) else None,
        })
    
    if not linhas:
        print("[AVISO] Nenhuma métrica válida para comparação agrupada.")
        out = pd.DataFrame(columns=[
            "Metrica","Controle_media","Controle_desvio",
            "Afasico_media","Afasico_desvio","MannWhitney_U","p_value"
        ])
    else:
        out = pd.DataFrame(linhas)
    
    garantir_dir(os.path.join(SAIDA_DIR, "analise_agrupada"))
    csv_path = os.path.join(SAIDA_DIR, "analise_agrupada", "inter_grupo_estatisticas_agrupadas.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Estatísticas inter-grupo agrupadas salvas: {csv_path}\n")
    
    return out


def comparacao_individual_pacientes(mets_ctrl: pd.DataFrame, mets_afa: pd.DataFrame) -> pd.DataFrame:
    """
    Gera comparações individuais entre cada paciente controle vs cada paciente afásico.
    Para cada par, calcula diferenças nas métricas principais por estímulo.
    """
    if mets_ctrl.empty or mets_afa.empty:
        print("[AVISO] Não há dados suficientes para comparação individual entre pacientes.")
        return pd.DataFrame()
    
    print(f"\n{'='*60}")
    print("Comparação INDIVIDUAL entre pacientes")
    print(f"{'='*60}")
    
    out_dir = os.path.join(SAIDA_DIR, "comparacoes_individuais")
    garantir_dir(out_dir)
    
    pacientes_ctrl = sorted(mets_ctrl["paciente_id"].unique())
    pacientes_afa = sorted(mets_afa["paciente_id"].unique())
    
    print(f"[INFO] Comparando {len(pacientes_ctrl)} controle(s) vs {len(pacientes_afa)} afásico(s)")
    
    comparacoes = []
    
    for pac_c in pacientes_ctrl:
        for pac_a in pacientes_afa:
            # Dados dos dois pacientes
            mets_c = mets_ctrl[mets_ctrl["paciente_id"] == pac_c]
            mets_a = mets_afa[mets_afa["paciente_id"] == pac_a]
            
            # Estímulos em comum
            stims_c = set(mets_c["Stimuli"].unique())
            stims_a = set(mets_a["Stimuli"].unique())
            stims_comum = sorted(stims_c.intersection(stims_a))
            
            if not stims_comum:
                continue
            
            for stim in stims_comum:
                m_c = mets_c[mets_c["Stimuli"] == stim].iloc[0]
                m_a = mets_a[mets_a["Stimuli"] == stim].iloc[0]
                
                # Calcula diferenças
                comparacoes.append({
                    "paciente_controle": pac_c,
                    "paciente_afasico": pac_a,
                    "Stimuli": stim,
                    "diff_tempo_resposta": m_c["tempo_resposta_s"] - m_a["tempo_resposta_s"] if not (pd.isna(m_c["tempo_resposta_s"]) or pd.isna(m_a["tempo_resposta_s"])) else np.nan,
                    "diff_n_sacadas": int(m_c["n_sacadas"] - m_a["n_sacadas"]),
                    "diff_n_fixacoes": int(m_c["n_fixacoes"] - m_a["n_fixacoes"]),
                    "diff_dur_fix": m_c["duracao_media_fix_s"] - m_a["duracao_media_fix_s"] if not (pd.isna(m_c["duracao_media_fix_s"]) or pd.isna(m_a["duracao_media_fix_s"])) else np.nan,
                    "diff_dispersao": m_c["dispersao_area"] - m_a["dispersao_area"],
                    "ctrl_tempo": m_c["tempo_resposta_s"],
                    "afa_tempo": m_a["tempo_resposta_s"],
                    "ctrl_sacadas": m_c["n_sacadas"],
                    "afa_sacadas": m_a["n_sacadas"],
                })
    
    if not comparacoes:
        print("[AVISO] Nenhuma comparação individual possível.")
        return pd.DataFrame()
    
    comp_df = pd.DataFrame(comparacoes)
    
    # Salvar CSV completo
    csv_path = os.path.join(out_dir, "comparacoes_individuais_detalhadas.csv")
    comp_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] Comparações individuais salvas: {csv_path}")
    
    # Resumo estatístico das diferenças
    resumo = []
    for col in ["diff_tempo_resposta", "diff_n_sacadas", "diff_n_fixacoes", "diff_dur_fix", "diff_dispersao"]:
        arr = comp_df[col].dropna().values
        if len(arr) > 0:
            resumo.append({
                "Metrica": col,
                "Media_diff": round(float(np.mean(arr)), 4),
                "Desvio_diff": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else np.nan,
                "Min_diff": round(float(np.min(arr)), 4),
                "Max_diff": round(float(np.max(arr)), 4),
            })
    
    resumo_df = pd.DataFrame(resumo)
    csv_resumo = os.path.join(out_dir, "resumo_comparacoes_individuais.csv")
    resumo_df.to_csv(csv_resumo, index=False, encoding="utf-8")
    print(f"[OK] Resumo de comparações salvo: {csv_resumo}\n")
    
    return comp_df


# =========================
# Execução principal
# =========================
def main():
    """
    Pipeline principal de análise.
    Gera análises em múltiplos níveis:
      1. Individual: cada paciente com seus testes
      2. Por grupo: análise segmentada por estímulo
      3. Agrupada: análise consolidada (todos os testes)
      4. Comparações: inter-grupo e individuais
    """
    print("\n" + "="*60)
    print("PIPELINE TROG-2 EYE-TRACKING ANALYSIS")
    print("Análises: Individual | Por Grupo | Agrupada | Comparações")
    print("="*60 + "\n")
    
    garantir_dir(SAIDA_DIR)

    # 1) Leitura e limpeza
    print("=" * 60)
    print("ETAPA 1: Carregamento de dados")
    print("=" * 60)
    df_ctrl = carregar_pasta_csv(PASTA_CONTROLE, "controle")
    df_afa  = carregar_pasta_csv(PASTA_AFASICO,  "afasico")
    
    df_ctrl = limpar_e_outliers(df_ctrl) if not df_ctrl.empty else df_ctrl
    df_afa  = limpar_e_outliers(df_afa)  if not df_afa.empty  else df_afa

    # 2) Processar por grupo SEGMENTADO (por estímulo) + INDIVIDUAL
    print("\n" + "=" * 60)
    print("ETAPA 2: Processamento SEGMENTADO + INDIVIDUAL")
    print("=" * 60)
    
    mets_ctrl = pd.DataFrame()
    mets_afa  = pd.DataFrame()

    if not df_ctrl.empty:
        _, mets_ctrl = processar_grupo_segmentado(df_ctrl, "controle")
    else:
        print("[AVISO] Grupo controle vazio após limpeza.")

    if not df_afa.empty:
        _, mets_afa  = processar_grupo_segmentado(df_afa,  "afasico")
    else:
        print("[AVISO] Grupo afásico vazio após limpeza.")

    # 3) Processar por grupo AGRUPADO (consolidado - todos os estímulos juntos)
    print("\n" + "=" * 60)
    print("ETAPA 3: Processamento AGRUPADO (consolidado)")
    print("=" * 60)
    
    mets_ctrl_agrup = pd.DataFrame()
    mets_afa_agrup = pd.DataFrame()
    
    if not df_ctrl.empty:
        mets_ctrl_agrup = processar_grupo_agrupado(df_ctrl, "controle")
    else:
        print("[AVISO] Grupo controle vazio para análise agrupada.")
    
    if not df_afa.empty:
        mets_afa_agrup = processar_grupo_agrupado(df_afa, "afasico")
    else:
        print("[AVISO] Grupo afásico vazio para análise agrupada.")

    # 4) Estatísticas inter-grupo SEGMENTADO
    print("\n" + "=" * 60)
    print("ETAPA 4: Estatísticas inter-grupo SEGMENTADO")
    print("=" * 60)
    
    if not (mets_ctrl.empty and mets_afa.empty):
        inter = estatisticas_inter_grupo(mets_ctrl, mets_afa)
        if not SCIPY_OK:
            print("[AVISO] SciPy não disponível — p-valores não calculados.")
    else:
        print("[AVISO] Não há métricas segmentadas para comparar entre os grupos.")

    # 5) Estatísticas inter-grupo AGRUPADO
    print("\n" + "=" * 60)
    print("ETAPA 5: Estatísticas inter-grupo AGRUPADO")
    print("=" * 60)
    
    if not (mets_ctrl_agrup.empty and mets_afa_agrup.empty):
        inter_agrup = estatisticas_inter_grupo_agrupado(mets_ctrl_agrup, mets_afa_agrup)
        if not SCIPY_OK:
            print("[AVISO] SciPy não disponível — p-valores não calculados.")
    else:
        print("[AVISO] Não há métricas agrupadas para comparar entre os grupos.")

    # 6) Comparações INDIVIDUAIS entre pacientes
    print("\n" + "=" * 60)
    print("ETAPA 6: Comparações INDIVIDUAIS entre pacientes")
    print("=" * 60)
    
    if not (mets_ctrl.empty or mets_afa.empty):
        comp_ind = comparacao_individual_pacientes(mets_ctrl, mets_afa)
    else:
        print("[AVISO] Não há dados suficientes para comparações individuais.")

    # 7) Resumos por grupo SEGMENTADO
    print("\n" + "=" * 60)
    print("ETAPA 7: Resumos por grupo SEGMENTADO")
    print("=" * 60)
    
    for grupo, mets in [("controle", mets_ctrl), ("afasico", mets_afa)]:
        if mets.empty:
            print(f"\n[{grupo.upper()}] Sem dados para resumo segmentado.")
            continue
        
        print(f"\n{'='*60}")
        print(f"RESUMO SEGMENTADO: {grupo.upper()}")
        print(f"{'='*60}")
        
        # Maior tempo médio de fixação
        if "duracao_media_fix_s" in mets.columns and mets["duracao_media_fix_s"].notna().any():
            r1 = mets.loc[mets["duracao_media_fix_s"].idxmax()]
            print(f"✓ Maior tempo médio de fixação: {r1['duracao_media_fix_s']:.4f}s")
            print(f"  Paciente: {r1['paciente_id']}, Estímulo: {r1['Stimuli']}")
        
        # Maior número de sacadas
        if "n_sacadas" in mets.columns and mets["n_sacadas"].notna().any():
            r2 = mets.loc[mets["n_sacadas"].idxmax()]
            print(f"✓ Maior nº de sacadas: {r2['n_sacadas']}")
            print(f"  Paciente: {r2['paciente_id']}, Estímulo: {r2['Stimuli']}")
        
        # Melhor/pior por tempo de resposta
        if "tempo_resposta_s" in mets.columns and mets["tempo_resposta_s"].notna().any():
            rb = mets.loc[mets["tempo_resposta_s"].idxmin()]
            rw = mets.loc[mets["tempo_resposta_s"].idxmax()]
            print(f"✓ Melhor tempo de resposta: {rb['tempo_resposta_s']:.3f}s")
            print(f"  Paciente: {rb['paciente_id']}, Estímulo: {rb['Stimuli']}")
            print(f"✓ Pior tempo de resposta: {rw['tempo_resposta_s']:.3f}s")
            print(f"  Paciente: {rw['paciente_id']}, Estímulo: {rw['Stimuli']}")

    # 8) Resumos por grupo AGRUPADO
    print("\n" + "=" * 60)
    print("ETAPA 8: Resumos por grupo AGRUPADO")
    print("=" * 60)
    
    for grupo, mets_agrup in [("controle", mets_ctrl_agrup), ("afasico", mets_afa_agrup)]:
        if mets_agrup.empty:
            print(f"\n[{grupo.upper()}] Sem dados para resumo agrupado.")
            continue
        
        print(f"\n{'='*60}")
        print(f"RESUMO AGRUPADO: {grupo.upper()}")
        print(f"{'='*60}")
        
        # Maior tempo médio de fixação
        if "duracao_media_fix_s" in mets_agrup.columns and mets_agrup["duracao_media_fix_s"].notna().any():
            r1 = mets_agrup.loc[mets_agrup["duracao_media_fix_s"].idxmax()]
            print(f"✓ Maior duração média de fixação: {r1['duracao_media_fix_s']:.4f}s")
            print(f"  Paciente: {r1['paciente_id']}")
        
        # Maior número total de sacadas
        if "total_sacadas" in mets_agrup.columns and mets_agrup["total_sacadas"].notna().any():
            r2 = mets_agrup.loc[mets_agrup["total_sacadas"].idxmax()]
            print(f"✓ Maior total de sacadas: {r2['total_sacadas']}")
            print(f"  Paciente: {r2['paciente_id']}")
        
        # Melhor/pior por tempo de resposta médio
        if "tempo_resposta_medio_s" in mets_agrup.columns and mets_agrup["tempo_resposta_medio_s"].notna().any():
            rb = mets_agrup.loc[mets_agrup["tempo_resposta_medio_s"].idxmin()]
            rw = mets_agrup.loc[mets_agrup["tempo_resposta_medio_s"].idxmax()]
            print(f"✓ Melhor tempo médio de resposta: {rb['tempo_resposta_medio_s']:.3f}s")
            print(f"  Paciente: {rb['paciente_id']}")
            print(f"✓ Pior tempo médio de resposta: {rw['tempo_resposta_medio_s']:.3f}s")
            print(f"  Paciente: {rw['paciente_id']}")
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"\nResultados salvos em: {os.path.abspath(SAIDA_DIR)}")
    print(f"\n{'='*60}")
    print("ESTRUTURA DE SAÍDA:")
    print("="*60)
    print(f"📁 {SAIDA_DIR}/")
    print(f"  ├── controle/")
    print(f"  │   ├── pacientes/              ← INDIVIDUAL (cada paciente)")
    print(f"  │   │   ├── [paciente_id]/")
    print(f"  │   │   │   ├── stim_1/, stim_2/, ... (gráficos por teste)")
    print(f"  │   │   │   ├── consolidado/         (gráficos todos os testes)")
    print(f"  │   │   │   └── [paciente]_metricas.csv")
    print(f"  │   ├── stim_1/, stim_2/, ...   ← POR GRUPO (todos os pacientes)")
    print(f"  │   │   └── [gráficos + CSV do grupo]")
    print(f"  │   ├── analise_agrupada/       ← AGRUPADA (consolidada)")
    print(f"  │   │   └── [gráficos + CSV consolidado]")
    print(f"  │   └── controle_concat.csv")
    print(f"  ├── afasico/")
    print(f"  │   └── [mesma estrutura]")
    print(f"  ├── comparacoes_individuais/    ← COMPARAÇÕES 1x1")
    print(f"  │   ├── comparacoes_individuais_detalhadas.csv")
    print(f"  │   └── resumo_comparacoes_individuais.csv")
    print(f"  ├── inter_grupo_estatisticas_por_estimulo.csv")
    print(f"  └── analise_agrupada/")
    print(f"      └── inter_grupo_estatisticas_agrupadas.csv")

if __name__ == "__main__":
    main()