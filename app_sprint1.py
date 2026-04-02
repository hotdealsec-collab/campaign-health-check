import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re

# --------------------------------------------------
# 1. ページ設定 & スタイル
# --------------------------------------------------
st.set_page_config(page_title="Piccoma Growth Audit Pro", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.small-note { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 2. 判定ロジック関数
# --------------------------------------------------
def safe_divide(a, b):
    return a / b if (pd.notna(a) and pd.notna(b) and b != 0) else np.nan

def map_score(value):
    score_map = {"良好": 100, "普通": 60, "注意": 30, "リスクあり": 20, "不明": 50}
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score): return "不明"
    if score >= 80: return "健全 (Healthy)"
    if score >= 60: return "観察 (Monitor)"
    if score >= 40: return "注意 (Warning)"
    return "要確認 (Critical)"

# --------------------------------------------------
# 3. データ処理エンジン (完全一致 & 重複排除)
# --------------------------------------------------
def run_growth_audit(df_adj, df_int, weights):
    # --- 1. 外部データ(Adjust)のクレンジングと集計 ---
    if 'campaign_network' not in df_adj.columns:
        return pd.DataFrame() 
        
    df_adj = df_adj.dropna(subset=['campaign_network']).copy()
    df_adj['campaign_network'] = df_adj['campaign_network'].astype(str).str.strip()
    
    if 'cohort_all_revenue' in df_adj.columns and 'all_revenue' not in df_adj.columns:
        df_adj.rename(columns={'cohort_all_revenue': 'all_revenue'}, inplace=True)
    
    def get_os_label(x):
        unique_os = sorted(x.dropna().unique().astype(str).tolist())
        if len(unique_os) > 1: return "Cross-platform"
        elif len(unique_os) == 1: return unique_os[0]
        else: return np.nan

    agg_dict = {
        'channel': lambda x: ', '.join(x.dropna().unique().astype(str)),
        'os_name': get_os_label
    }
    if 'cost' in df_adj.columns: agg_dict['cost'] = 'sum'
    else: df_adj['cost'] = 0; agg_dict['cost'] = 'sum'
    
    if 'installs' in df_adj.columns: agg_dict['installs'] = 'sum'
    else: df_adj['installs'] = 0; agg_dict['installs'] = 'sum'
    
    if 'reattributions' in df_adj.columns: agg_dict['reattributions'] = 'sum'
    if 'skad_installs' in df_adj.columns: agg_dict['skad_installs'] = 'sum'
    if 'all_revenue' in df_adj.columns: agg_dict['all_revenue'] = 'sum'

    adj_grouped = df_adj.groupby('campaign_network').agg(agg_dict).reset_index()

    # --- 2. 内部データのクレンジングと集計 ---
    if 'campaign_name' not in df_int.columns:
        return pd.DataFrame()
        
    df_int = df_int.dropna(subset=['campaign_name']).copy()
    def clean_campaign_name(name):
        return re.sub(r'\s*\(\d+\)\s*$', '', str(name)).strip()

    df_int['campaign_name_clean'] = df_int['campaign_name'].apply(clean_campaign_name)
    int_grouped = df_int.groupby('campaign_name_clean').agg({
        'user_count': 'sum', 'ru_count': 'sum', 'd1_count': 'sum',
        'd7_count': 'sum', 'product_count': 'sum', 'bm_user_count': 'sum', 'r_sales': 'sum'
    }).reset_index()

    # --- 3. 結合 (Adjust基準 Left Join) ---
    df = pd.merge(adj_grouped, int_grouped, left_on='campaign_network', right_on='campaign_name_clean', how='left')
    if df.empty: return df

    # --- ノイズキャンペーンの除外 ---
    total_installs = df['installs'] + df.get('reattributions', 0)
    df = df[~((total_installs < 30) & (df['cost'] == 0))].copy()
    
    if df.empty: return df

    # --- 4. 指標計算とスコアリング ---
    df["cpi"] = df.apply(lambda x: safe_divide(x["cost"], x["installs"] + x.get("reattributions", 0)), axis=1)
    df["activation"] = df.apply(lambda x: safe_divide(x["ru_count"], x["user_count"]), axis=1)
    df["intensity"] = df.apply(lambda x: safe_divide(x["product_count"], x["ru_count"]), axis=1)
    df["retention_d7"] = df.apply(lambda x: safe_divide(x["d7_count"], x["ru_count"]), axis=1)
    df["bm_rate"] = df.apply(lambda x: safe_divide(x["bm_user_count"], x["user_count"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["r_sales"], x["user_count"]), axis=1)
    df["payback"] = df.apply(lambda x: safe_divide(x["cost"], x["r_sales"]), axis=1)

    avg_cpi = df["cpi"].mean()
    avg_int = df["intensity"].mean()
    avg_bm = df["bm_rate"].mean()

    # [UPDATE] Traffic Score: Cost가 0이고 ARPU가 10 이상이면 연동 오류로 간주하여 50점(不明) 부여
    def get_traffic_score(row):
        if row["cost"] == 0:
            return 50 if row.get("arpu", 0) >= 10 else 0  
        if pd.isna(row["cpi"]) or pd.isna(avg_cpi):
            return 50
        if row["cpi"] <= avg_cpi * 0.85:
            return 100
        if row["cpi"] <= avg_cpi * 1.15:
            return 60
        return 30
        
    df["s_traffic"] = df.apply(get_traffic_score, axis=1)
    df["s_activation"] = df["activation"].apply(lambda x: 50 if pd.isna(x) else (100 if x >= 0.7 else (60 if x >= 0.5 else 30)))
    df["s_intensity"] = df["intensity"].apply(lambda x: "不明" if pd.isna(x) or pd.isna(avg_int) else ("良好" if x >= avg_int*1.15 else ("普通" if x >= avg_int*0.85 else "注意"))).map(map_score)
    df["s_retention"] = df["retention_d7"].apply(lambda x: 50 if pd.isna(x) else (100 if x >= 0.25 else (60 if x >= 0.15 else 30)))
    df["s_bm"] = df["bm_rate"].apply(lambda x: "不明" if pd.isna(x) or pd.isna(avg_bm) else ("良好" if x >= avg_bm*1.15 else ("普通" if x >= avg_bm*0.85 else "注意"))).map(map_score)
    
    # [UPDATE] Payback Score: Cost가 0이고 ARPU가 10 이상이면 연동 오류로 간주하여 50점(不明) 부여
    df["s_payback"] = df.apply(
        lambda x: (50 if x.get("arpu", 0) >= 10 else 0) if x["cost"] == 0 else (50 if pd.isna(x["payback"]) else (100 if x["payback"] <= 1.2 else (60 if x["payback"] <= 2.5 else 20))), axis=1
    )

    df["growth_health_score"] = (
        df["s_traffic"] * (weights['traffic'] / 100) + 
        df["s_activation"] * (weights['activation'] / 100) + 
        df["s_intensity"] * (weights['intensity'] / 100) + 
        df["s_retention"] * (weights['retention'] / 100) + 
        df["s_bm"] * (weights['bm'] / 100) + 
        df["s_payback"] * (weights['payback'] / 100)
    ).round(1)
    
    df["growth_category"] = df["growth_health_score"].apply(score_category)
