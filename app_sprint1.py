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
# 2. 判定ロジック関数 (Continuous Scoring Helpers)
# --------------------------------------------------
def safe_divide(a, b):
    return a / b if (pd.notna(a) and pd.notna(b) and b != 0) else np.nan

def score_category(score):
    if pd.isna(score): return "不明"
    if score >= 80: return "健全 (Healthy)"
    if score >= 60: return "観察 (Monitor)"
    if score >= 40: return "注意 (Warning)"
    return "要確認 (Critical)"

def calc_continuous_score(val, threshold_excellent, threshold_good):
    if pd.isna(val): return 50.0
    if threshold_excellent == threshold_good: return 50.0
    if val >= threshold_excellent: return 100.0
    if val >= threshold_good:
        return 60.0 + ((val - threshold_good) / (threshold_excellent - threshold_good)) * 40.0
    if threshold_good == 0: return 0.0
    return max(0.0, (val / threshold_good) * 60.0)

def calc_continuous_score_inverse(val, threshold_excellent, threshold_good):
    if pd.isna(val): return 50.0
    if threshold_excellent == threshold_good: return 50.0
    if val <= threshold_excellent: return 100.0
    if val <= threshold_good:
        return 60.0 + ((threshold_good - val) / (threshold_good - threshold_excellent)) * 40.0
    if threshold_good == 0: return 0.0
    return max(0.0, 60.0 - ((val - threshold_good) / threshold_good) * 60.0)

# --------------------------------------------------
# 3. データ処理エンジン
# --------------------------------------------------
def run_growth_audit(df_adj, df_int, weights):
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

    # [NEW] 캠페인 타입 정의: reattributions 1건 이상이면 '復帰', 아니면 '新規'
    adj_grouped['campaign_type'] = adj_grouped['reattributions'].apply(lambda x: "復帰" if x >= 1 else "新規")

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

    df = pd.merge(adj_grouped, int_grouped, left_on='campaign_network', right_on='campaign_name_clean', how='left')
    if df.empty: return df

    # --- 獲得ボリュームの算出とノイズ除外 ---
    df['total_installs'] = df['installs'] + df.get('reattributions', 0)
    df = df[~((df['total_installs'] < 30) & (df['cost'] == 0))].copy()
    
    if df.empty: return df

    # --- 4. 指標計算 ---
    df["cpi"] = df.apply(lambda x: safe_divide(x["cost"], x["total_installs"]), axis=1)
    df["activation"] = df.apply(lambda x: safe_divide(x["ru_count"], x["user_count"]), axis=1)
    df["intensity"] = df.apply(lambda x: safe_divide(x["product_count"], x["ru_count"]), axis=1)
    df["retention_d7"] = df.apply(lambda x: safe_divide(x["d7_count"], x["user_count"]), axis=1)
    df["bm_rate"] = df.apply(lambda x: safe_divide(x["bm_user_count"], x["user_count"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["r_sales"], x["user_count"]), axis=1)
    df["payback"] = df.apply(lambda x: safe_divide(x["r_sales"], x["cost"]), axis=1) # ROAS

    avg_cpi = df["cpi"].mean()
    avg_int = df["intensity"].mean()
    avg_ret = df["retention_d7"].mean()
    avg_bm = df["bm_rate"].mean()

    # --- 5. スコアリング ---
    df["s_volume"] = df["total_installs"].rank(pct=True) * 100

    def get_traffic_score(row):
        if row["cost"] == 0:
            return 50 if row.get("arpu", 0) >= 10 else 0  
        if pd.isna(row["cpi"]) or pd.isna(avg_cpi) or avg_cpi == 0:
            return 50
        return calc_continuous_score_inverse(row["cpi"], avg_cpi * 0.85, avg_cpi * 1.15)
    df["s_traffic"] = df.apply(get_traffic_score, axis=1)

    df["s_activation"] = df["activation"].apply(lambda x: calc_continuous_score(x, 0.70, 0.50))
    df["s_intensity"] = df["intensity"].apply(lambda x: calc_continuous_score(x, avg_int * 1.15, avg_int * 0.85) if avg_int > 0 else 50)
    df["s_retention"] = df["retention_d7"].apply(lambda x: calc_continuous_score(x, avg_ret * 1.15, avg_ret * 0.85) if avg_ret > 0 else 50)
    df["s_bm"] = df["bm_rate"].apply(lambda x: calc_continuous_score(x, avg_bm * 1.15, avg_bm * 0.85) if avg_bm > 0 else 50)
    
    def get_payback_score(row):
        if row["cost"] == 0:
            return 50 if row.get("arpu", 0) >= 10 else 0
        return calc_continuous_score(row["payback"], 0.80, 0.40)
    df["s_payback"] = df.apply(get_payback_score, axis=1)

    # --- 総合スコア算出 ---
    df["growth_health_score"] = (
        df["s_volume"] * (weights['volume'] / 100) +
        df["s_traffic"] * (weights['traffic'] / 100) + 
        df["s_activation"] * (weights['activation'] / 100) + 
        df["s_intensity"] * (weights['intensity'] / 100) + 
        df["s_retention"] * (weights['retention'] / 100) + 
        df["s_bm"] * (weights['bm'] / 100) + 
        df["s_payback"] * (weights['payback'] / 100)
    ).round(1)
    
    df["growth_category"] = df["growth_health_score"].apply(score_category)
    
    def calculate_confidence(row):
        score = 100
        if row["cost"] == 0:
            if row.get("arpu", 0) >= 10:
                score -= 20
            else:
                score -= 50
        if pd.isna(row.get("user_count")): score -= 50
        if pd.notna(row.get("skad_installs")) and row["skad_installs"] > 0: score -= 20
        return max(score, 0)

    df["confidence_score"] = df.apply(calculate_confidence, axis=1)
    
    return df

# --------------------------------------------------
# 4. メイン UI
# --------------------------------------------------
st.title("Campaign Health Check Ver3")

st.sidebar.header("1. Upload Data")
adj_file = st.sidebar.file_uploader("Adjust CSV", type="csv")
int_file = st.sidebar.file_uploader("Internal SQL CSV", type="csv")

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ スコアの計算ロジック（Guide）", expanded=False):
    st.markdown("""
    **📈 Growth Health Score (0~100点)**
    * **Volume**: 獲得規模のパーセンタイル評価
    * **Campaign Type**: Reattributionsが1以上の場合は'復帰'、0の場合は'新規'と分類
    * **기타**: 데이터 기반 연속형 스코어 적용
    """)

st.sidebar.header("2. Weight Settings (%)")
w_volume = st.sidebar.slider("Volume (獲得ボリューム)", min_value=0, max_value=100, value=10, step=5)
w_traffic = st.sidebar.slider("Traffic (CPI効率)", min_value=0, max_value=100, value=10, step=5)
w_activation = st.sidebar.slider("Activation (作品閲覧転換率)", min_value=0, max_value=100, value=10, step=5)
w_intensity = st.sidebar.slider("Intensity (平均閲覧作品数)", min_value=0, max_value=100, value=15, step=5)
w_retention = st.sidebar.slider("Retention (D7維持率)", min_value=0, max_value=100, value=20, step=5)
w_bm = st.sidebar.slider("BM Contribution (BM利用率)", min_value=0, max_value=100, value=15, step=5)
w_payback = st.sidebar.slider("Payback (投資回収効率/ROAS)", min_value=0, max_value=100, value=20, step=5)

weights_dict = {'volume': w_volume, 'traffic': w_traffic, 'activation': w_activation, 'intensity': w_intensity, 'retention': w_retention, 'bm': w_bm, 'payback': w_payback}

if adj_file and int_file:
    audit_df = run_growth_audit(pd.read_csv(adj_file), pd.read_csv(int_file), weights_dict)

    if not audit_df.empty:
        st.markdown("### Filters")
        f1, f2, f3, f4, f5 = st.columns(5)
        
        # [NEW] 캠페인 타입 필터 추가
        type_opts = sorted(audit_df['campaign_type'].unique().tolist())
        sel_type = f1.multiselect("Campaign Type", type_opts, placeholder="All (New/Return)")
        
        channel_opts = sorted(audit_df['channel'].dropna().unique().tolist())
        sel_ch = f2.multiselect("Channel", channel_opts, placeholder="All")
        
        os_opts = sorted(audit_df['os_name'].dropna().unique().tolist())
        sel_os = f3.selectbox("OS", ["All"] + os_opts)
        
        category_opts = sorted(audit_df['growth_category'].dropna().unique().tolist())
        sel_ct = f4.selectbox("Category", ["All"] + category_opts)

        f_df = audit_df.copy()
        if sel_type: f_df = f_df[f_df['campaign_type'].isin(sel_type)]
        if sel_ch: f_df = f_df[f_df['channel'].isin(sel_ch)]
        if sel_os != "All": f_df = f_df[f_df['os_name'] == sel_os]
        if sel_ct != "All": f_df = f_df[f_df['growth_category'] == sel_ct]

        f_df["ranking_score"] = (f_df["growth_health_score"] * (f_df["confidence_score"] / 100.0)).round(1)
        f_df = f_df.sort_values(by=["ranking_score", "growth_health_score"], ascending=[False, False]).reset_index(drop=True)
        f_df.insert(0, 'Rank', range(1, len(f_df) + 1))

        st.markdown("### Campaign Table")
        
        # [NEW] 캠페인 타입 컬럼을 테이블에 추가
        display_cols = [
            "Rank", "ranking_score", "campaign_network", "campaign_type", "channel", "os_name", "growth_category", "growth_health_score", 
            "total_installs", "cpi", "activation", "intensity", "retention_d7", "bm_rate", "arpu", "payback"
        ]

        st.dataframe(
            f_df[display_cols].style
            .format({
                "ranking_score": "{:.1f}", "total_installs": "{:,.0f}", "cpi": "{:.2f}", 
                "activation": "{:.1%}", "intensity": "{:.2f}", "retention_d7": "{:.1%}", 
                "bm_rate": "{:.1%}", "arpu": "{:,.0f}", "payback": "{:.1%}"
            }, na_rep="N/A"), 
            use_container_width=True, height=500, hide_index=True
        )
else:
    st.info("左右のCSVファイルをアップロードしてください。")
