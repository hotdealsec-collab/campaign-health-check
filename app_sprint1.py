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

    # [NEW] Traffic Score: Cost가 0이더라도 ARPU가 10 이상이면 100점 부여
    def get_traffic_score(row):
        if row["cost"] == 0:
            return 100 if row.get("arpu", 0) >= 10 else 0  
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
    
    # [NEW] Payback Score: Cost가 0이더라도 ARPU가 10 이상이면 100점 부여 (무료로 수익 창출)
    df["s_payback"] = df.apply(
        lambda x: (100 if x.get("arpu", 0) >= 10 else 0) if x["cost"] == 0 else (50 if pd.isna(x["payback"]) else (100 if x["payback"] <= 1.2 else (60 if x["payback"] <= 2.5 else 20))), axis=1
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
    
    # [NEW] Confidence Score: ARPU가 10 이상이면 실질 가치가 증명되었으므로 페널티 완화 (-50 -> -20)
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
st.title("Campaign Health Check Ver.2")

st.sidebar.header("1. Upload Data")
adj_file = st.sidebar.file_uploader("Adjust CSV", type="csv")
int_file = st.sidebar.file_uploader("Internal SQL CSV", type="csv")

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ スコアの計算ロジック（Guide）", expanded=False):
    st.markdown("""
    **📈 Growth Health Score (0~100点)**
    各指標を基準値（平均値や絶対値）と比較し、「良好(100点)」「普通(60点)」「注意(30点)」等にスコア化。それに以下のスライダーの重みを掛けて合算します。
    ※ 有効なデータのみを分析するため、(Installs + Reattributions) が30未満かつコストが0のノイズキャンペーンは自動的に除外されます。
    ※ 広告コストが0の場合、原則としてTrafficおよびPaybackスコアは0点処理されますが、**ARPU（ユーザー平均単価）が10以上の場合は「優良な無料流入」とみなし、両スコアに100点が付与**されます。
    
    **📊 Confidence Score (0~100点)**
    データの信頼度を表します。基本100点から、以下の要因で減点されます。
    * **-50点**: 広告コスト(Cost)が 0 の場合（※ただし、ARPUが10以上の優良キャンペーンは実質価値が証明されているため **-20点** に軽減）
    * **-50点**: 社内データが紐付かない場合（内部行動分析不可）
    * **-20点**: iOSのSKANデータが含まれる場合（乖離リスクあり）
    
    **🏆 Ranking (順位付けの基準)**
    表の順位は、単なるHealth Scoreではなく **[ Health Score × (Confidence Score / 100) ]** の「信頼度調整後スコア」を用いて決定されます。
    """)

st.sidebar.header("2. Weight Settings (%)")
st.sidebar.markdown("<div class='small-note'>各指標の重みを調整できます（合計100%推奨）</div>", unsafe_allow_html=True)

w_traffic = st.sidebar.slider("Traffic (CPI効率)", min_value=0, max_value=100, value=10, step=5, help="インストールあたりの獲得コスト効率（CPI）を評価します。")
w_activation = st.sidebar.slider("Activation (作品閲覧転換率)", min_value=0, max_value=100, value=15, step=5, help="インストール後、実際に作品を閲覧したユーザーの割合です。")
w_intensity = st.sidebar.slider("Intensity (平均閲覧作品数)", min_value=0, max_value=100, value=15, step=5, help="1ユーザーあたりの平均閲覧作品数で、エンゲージメントの深さを測ります。")
w_retention = st.sidebar.slider("Retention (D7維持率)", min_value=0, max_value=100, value=20, step=5, help="インストールから7日後もアプリを利用しているユーザーの割合です。")
w_bm = st.sidebar.slider("BM Contribution (BM利用率)", min_value=0, max_value=100, value=25, step=5, help="ビジネスモデル（課金など）に貢献したユーザーの割合です。")
w_payback = st.sidebar.slider("Payback (投資回収効率)", min_value=0, max_value=100, value=15, step=5, help="投下した広告費に対する売上の回収効率（ROAS等）を評価します。")

total_weight = w_traffic + w_activation + w_intensity + w_retention + w_bm + w_payback

if total_weight != 100:
    st.sidebar.warning(f"⚠️ 現在の合計は {total_weight}% です。正確な評価のため 100% に合わせてください。")
else:
    st.sidebar.success(f"✅ 合計 100% (最適)")

weights_dict = {
    'traffic': w_traffic,
    'activation': w_activation,
    'intensity': w_intensity,
    'retention': w_retention,
    'bm': w_bm,
    'payback': w_payback
}

if adj_file and int_file:
    audit_df = run_growth_audit(pd.read_csv(adj_file), pd.read_csv(int_file), weights_dict)

    if audit_df.empty:
        st.error("❌ キャンペーンの一致が確認できませんでした。Adjustの'campaign_network'と社内データの'campaign_name'を確認してください。")
    else:
        # --- Filters ---
        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns(4)
        
        channel_opts = sorted(audit_df['channel'].dropna().unique().tolist())
        sel_ch = f1.multiselect("Channel", channel_opts, placeholder="All (Select to filter)")
        
        os_opts = sorted(audit_df['os_name'].dropna().unique().tolist())
        sel_os = f2.selectbox("OS", ["All"] + os_opts)
        
        campaign_opts = sorted(audit_df['campaign_network'].dropna().unique().tolist())
        sel_cp = f3.multiselect("Campaign", campaign_opts, placeholder="All (Select to filter)")
        
        category_opts = sorted(audit_df['growth_category'].dropna().unique().tolist())
        sel_ct = f4.selectbox("Growth Category", ["All"] + category_opts)

        f_df = audit_df.copy()
        if sel_ch: f_df = f_df[f_df['channel'].isin(sel_ch)]
        if sel_os != "All": f_df = f_df[f_df['os_name'] == sel_os]
        if sel_cp: f_df = f_df[f_df['campaign_network'].isin(sel_cp)]
        if sel_ct != "All": f_df = f_df[f_df['growth_category'] == sel_ct]

        # --- Data Sorting & Ranking ---
        f_df["ranking_score"] = f_df["growth_health_score"] * (f_df["confidence_score"] / 100.0)
        f_df = f_df.sort_values(by=["ranking_score", "growth_health_score"], ascending=[False, False]).reset_index(drop=True)
        f_df.insert(0, 'Rank', range(1, len(f_df) + 1))

        # --- Overview ---
        st.markdown("### Overview")
        k1, k2, k3 = st.columns(3)
        if len(f_df) > 0:
            mean_score = f_df['growth_health_score'].mean()
            mean_conf = f_df['confidence_score'].mean()
            k1.metric("Average Growth Score", f"{mean_score:.1f}" if pd.notna(mean_score) else "N/A")
            k2.metric("Average Confidence", f"{mean_conf:.1f}" if pd.notna(mean_conf) else "N/A")
            k3.metric("Campaigns (Unique)", len(f_df))
        else:
            k1.metric("Average Growth Score", "N/A")
            k2.metric("Average Confidence", "N/A")
            k3.metric("Campaigns (Unique)", 0)

        # --- Positioning Chart ---
        st.markdown("### Campaign Positioning")
        f_df_plot = f_df.copy()
        np.random.seed(42) 
        f_df_plot["plot_x"] = f_df_plot["growth_health_score"] + np.random.uniform(-1.0, 1.0, len(f_df_plot))
        f_df_plot["plot_y"] = f_df_plot["confidence_score"] + np.random.uniform(-1.0, 1.0, len(f_df_plot))

        scatter = alt.Chart(f_df_plot).mark_circle(size=140, opacity=0.7).encode(
            x=alt.X("plot_x:Q", title="Growth Health Score", scale=alt.Scale(zero=False)),
            y=alt.Y("plot_y:Q", title="Confidence", scale=alt.Scale(zero=False)),
            color=alt.Color("growth_category:N", title="Category"),
            tooltip=["Rank", "campaign_network", "growth_health_score", "confidence_score", "growth_category"]
        ).properties(height=400).interactive()
        st.altair_chart(scatter, use_container_width=True)

        # --- Campaign Table & Download CSV ---
        col_title, col_btn = st.columns([4, 1])
        col_title.markdown("### Campaign Table")
        
        display_cols = [
            "Rank", "campaign_network", "channel", "os_name", "growth_category", "growth_health_score", "confidence_score",
            "cpi", "activation", "intensity", "retention_d7", "bm_rate", "arpu", "payback"
        ]

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8-sig')

        csv_data = convert_df(f_df[display_cols])
        col_btn.download_button(label="📥 Download CSV", data=csv_data, file_name='campaign_health_check_ranked.csv', mime='text/csv')

        def style_red(val):
            return "background-color: rgba(239, 68, 68, 0.2); color: #ef4444;" if isinstance(val, (int, float)) and val < 60 else ""
        
        st.dataframe(
            f_df[display_cols].style
            .map(style_red, subset=["growth_health_score"])
            .format({
                "cpi": "{:.2f}", 
                "activation": "{:.1%}", 
                "intensity": "{:.2f}", 
                "retention_d7": "{:.1%}", 
                "bm_rate": "{:.1%}", 
                "arpu": "{:,.0f}", 
                "payback": "{:.2f}"
            }, na_rep="N/A"), 
            use_container_width=True, height=500, hide_index=True
        )
else:
    st.info("左右のCSVファイルをアップロードしてください。")
