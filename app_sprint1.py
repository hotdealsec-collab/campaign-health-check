import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Campaign Health Check",
    layout="wide"
)

# --------------------------------------------------
# Style
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.9rem;
}
.mono-box {
    background-color: #111827;
    color: #f9fafb;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    font-family: monospace;
    font-size: 0.92rem;
    line-height: 1.8;
}
.section-caption {
    color: #9ca3af;
    font-size: 0.88rem;
    margin-top: -0.4rem;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Required columns
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "channel",
    "campaign",
    "os",
    "spend",
    "installs",
    "activated_users",
    "d1_retention",
    "d3_retention",
    "d7_retention",
    "revenue",
    "skan_only",
    "strategic_channel",
    "period_start",
    "period_end",
]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def normalize_bool(value):
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in ["true", "1", "yes", "y"]

def rate_relative_low_is_good(value, avg_value):
    if pd.isna(value) or pd.isna(avg_value):
        return "不明"
    if value <= avg_value * 0.85:
        return "良好"
    elif value <= avg_value * 1.15:
        return "普通"
    return "注意"

def rate_relative_high_is_good(value, avg_value):
    if pd.isna(value) or pd.isna(avg_value):
        return "不明"
    if value >= avg_value * 1.15:
        return "良好"
    elif value >= avg_value * 0.85:
        return "普通"
    return "注意"

def map_score(value):
    score_map = {
        "良好": 100,
        "普通": 60,
        "注意": 30,
        "リスクあり": 20,
        "不明": 50,
    }
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score):
        return "不明"
    if score >= 80:
        return "健全"
    if score >= 60:
        return "観察"
    if score >= 40:
        return "注意"
    return "要確認"

def measurement_confidence_level(score):
    if pd.isna(score):
        return "不明"
    if score >= 80:
        return "高"
    if score >= 60:
        return "中"
    if score >= 40:
        return "低"
    return "極めて低"

# --------------------------------------------------
# Core engine
# --------------------------------------------------
def run_growth_audit_lite(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必須カラムが不足しています: {missing_cols}")

    numeric_cols = [
        "spend", "installs", "activated_users",
        "d1_retention", "d3_retention", "d7_retention", "revenue",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["d1_retention", "d3_retention", "d7_retention"]:
        df[col] = df[col].clip(lower=0, upper=0.9999)

    df["skan_only"] = df["skan_only"].apply(normalize_bool)
    df["strategic_channel"] = df["strategic_channel"].apply(normalize_bool)
    df["os"] = df["os"].astype(str).str.strip()

    df["cpi"] = df.apply(lambda x: safe_divide(x["spend"], x["installs"]), axis=1)
    df["activation_rate"] = df.apply(lambda x: safe_divide(x["activated_users"], x["installs"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["revenue"], x["installs"]), axis=1)

    df["early_signal_score"] = (
        df["d1_retention"] * 0.2 +
        df["d3_retention"] * 0.3 +
        df["d7_retention"] * 0.5
    ) * 100

    def retention_curve_quality(row):
        if pd.isna(row["d1_retention"]) or pd.isna(row["d7_retention"]) or row["d1_retention"] <= 0:
            return np.nan
        return row["d7_retention"] / row["d1_retention"]

    df["retention_curve_quality"] = df.apply(retention_curve_quality, axis=1)

    def estimate_payback(cpi, arpu):
        if pd.isna(cpi) or pd.isna(arpu) or arpu <= 0:
            return np.nan
        return cpi / arpu

    df["payback_period"] = df.apply(lambda x: estimate_payback(x["cpi"], x["arpu"]), axis=1)

    avg_cpi = df["cpi"].mean(skipna=True)
    avg_arpu = df["arpu"].mean(skipna=True)

    def score_activation_efficiency(activation_rate):
        if pd.isna(activation_rate):
            return "不明"
        if activation_rate >= 0.50:
            return "良好"
        elif activation_rate >= 0.30:
            return "普通"
        return "注意"

    def score_retention_stability(d7_retention):
        if pd.isna(d7_retention):
            return "不明"
        if d7_retention >= 0.25:
            return "良好"
        elif d7_retention >= 0.15:
            return "普通"
        return "注意"

    def score_early_signal(early_signal_score):
        if pd.isna(early_signal_score):
            return "不明"
        if early_signal_score >= 30:
            return "良好"
        elif early_signal_score >= 18:
            return "普通"
        return "注意"

    def score_curve_quality(curve_quality):
        if pd.isna(curve_quality):
            return "不明"
        if curve_quality >= 0.45:
            return "良好"
        elif curve_quality >= 0.30:
            return "普通"
        return "注意"

    def score_payback_health(payback_period):
        if pd.isna(payback_period):
            return "不明"
        if payback_period <= 1.5:
            return "良好"
        elif payback_period <= 3.0:
            return "普通"
        return "リスクあり"

    df["traffic_efficiency"] = df["cpi"].apply(lambda x: rate_relative_low_is_good(x, avg_cpi))
    df["activation_efficiency"] = df["activation_rate"].apply(score_activation_efficiency)
    df["retention_stability"] = df["d7_retention"].apply(score_retention_stability)
    df["early_signal_health"] = df["early_signal_score"].apply(score_early_signal)
    df["curve_quality_health"] = df["retention_curve_quality"].apply(score_curve_quality)
    df["revenue_efficiency"] = df["arpu"].apply(lambda x: rate_relative_high_is_good(x, avg_arpu))
    df["payback_health"] = df["payback_period"].apply(score_payback_health)

    df["traffic_score"] = df["traffic_efficiency"].map(map_score)
    df["activation_score"] = df["activation_efficiency"].map(map_score)
    df["early_signal_score_norm"] = df["early_signal_health"].map(map_score)
    df["retention_score"] = df["retention_stability"].map(map_score)
    df["revenue_score"] = df["revenue_efficiency"].map(map_score)
    df["payback_score"] = df["payback_health"].map(map_score)

    df["growth_health_score"] = (
        df["traffic_score"] * 0.10 +
        df["activation_score"] * 0.15 +
        df["early_signal_score_norm"] * 0.20 +
        df["retention_score"] * 0.20 +
        df["revenue_score"] * 0.20 +
        df["payback_score"] * 0.15
    ).round(1)

    df["growth_health_category"] = df["growth_health_score"].apply(score_category)

    def measurement_confidence_score(row):
        score = 100
        if pd.isna(row["spend"]) or row["spend"] == 0:
            score -= 50
        if pd.isna(row["installs"]) or row["installs"] == 0:
            score -= 30
        if str(row["os"]).strip().lower() == "ios":
            score -= 15
        if row["skan_only"]:
            score -= 20
        return max(score, 0)

    df["measurement_confidence_score"] = df.apply(measurement_confidence_score, axis=1)
    df["measurement_confidence_level"] = df["measurement_confidence_score"].apply(measurement_confidence_level)

    # Simple zone only for visual grouping
    def simple_zone(row):
        if row["growth_health_score"] >= 75 and row["measurement_confidence_score"] >= 60:
            return "Watchlist+"   # actually good candidates, but toned down wording
        if row["growth_health_score"] < 50 and row["measurement_confidence_score"] >= 60:
            return "Check"
        if row["measurement_confidence_score"] < 40:
            return "Low Confidence"
        return "Monitor"

    df["simple_zone"] = df.apply(simple_zone, axis=1)

    summary = {
        "avg_growth_health_score": df["growth_health_score"].mean(skipna=True),
        "avg_measurement_confidence_score": df["measurement_confidence_score"].mean(skipna=True),
        "campaign_count": len(df),
    }

    summary_df = pd.DataFrame([summary])

    return df, summary_df

# --------------------------------------------------
# Table styling
# --------------------------------------------------
def highlight_growth_score(val):
    if pd.isna(val):
        return ""
    if val < 50:
        return "background-color: rgba(239, 68, 68, 0.35); color: white;"
    if val < 60:
        return "background-color: rgba(245, 158, 11, 0.30); color: white;"
    return ""

def highlight_measurement_score(val):
    if pd.isna(val):
        return ""
    if val < 40:
        return "background-color: rgba(220, 38, 38, 0.38); color: white;"
    if val < 60:
        return "background-color: rgba(234, 179, 8, 0.30); color: white;"
    return ""

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("## Campaign Health Check")
st.markdown(
    "<div class='small-note'>Performance × Measurement Confidence</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.markdown("## Upload")
st.sidebar.markdown("### Required columns")
st.sidebar.markdown(
    f"""
    <div class="mono-box">
    {"<br>".join(REQUIRED_COLUMNS)}
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("CSV must include the required columns exactly.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("CSVをアップロードすると、簡易ヘルスチェック画面が表示されます。")
else:
    try:
        raw_df = pd.read_csv(uploaded_file)
        audit_df, summary_df = run_growth_audit_lite(raw_df.copy())

        # Top KPIs - only 3
        st.markdown("### Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Growth Score", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
        c2.metric("Average Measurement Confidence", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
        c3.metric("Campaigns", int(summary_df.loc[0, "campaign_count"]))

        st.markdown("### Filters")
        f1, f2, f3 = st.columns(3)

        channel_options = ["All"] + sorted(audit_df["channel"].dropna().unique().tolist())
        campaign_options = ["All"] + sorted(audit_df["campaign"].dropna().unique().tolist())
        os_options = ["All"] + sorted(audit_df["os"].dropna().unique().tolist())

        selected_channel = f1.selectbox("Channel", channel_options)
        selected_campaign = f2.selectbox("Campaign", campaign_options)
        selected_os = f3.selectbox("OS", os_options)

        filtered_df = audit_df.copy()
        if selected_channel != "All":
            filtered_df = filtered_df[filtered_df["channel"] == selected_channel]
        if selected_campaign != "All":
            filtered_df = filtered_df[filtered_df["campaign"] == selected_campaign]
        if selected_os != "All":
            filtered_df = filtered_df[filtered_df["os"] == selected_os]

        st.markdown("### Strategic Scatter")
        st.markdown(
            "<div class='section-caption'>Quick visual positioning for campaign monitoring.</div>",
            unsafe_allow_html=True
        )

        scatter_df = filtered_df.copy()
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("growth_health_score:Q", title="Growth Health Score"),
                y=alt.Y("measurement_confidence_score:Q", title="Measurement Confidence Score"),
                color=alt.Color("simple_zone:N", title="View"),
                tooltip=[
                    alt.Tooltip("channel:N", title="Channel"),
                    alt.Tooltip("campaign:N", title="Campaign"),
                    alt.Tooltip("os:N", title="OS"),
                    alt.Tooltip("growth_health_score:Q", title="Growth Score", format=".1f"),
                    alt.Tooltip("measurement_confidence_score:Q", title="Measurement Score", format=".1f"),
                    alt.Tooltip("simple_zone:N", title="Zone"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

        st.markdown("### Campaign Table")
        view_columns = [
            "channel", "campaign", "os",
            "spend", "installs",
            "growth_health_score", "measurement_confidence_score",
            "growth_health_category", "measurement_confidence_level"
        ]

        styled_df = (
            filtered_df[view_columns]
            .style
            .applymap(highlight_growth_score, subset=["growth_health_score"])
            .applymap(highlight_measurement_score, subset=["measurement_confidence_score"])
        )

        st.dataframe(styled_df, use_container_width=True, height=460)

        st.markdown("### Notes")
        st.caption(
            "This view is intended for lightweight campaign monitoring and hypothesis checking."
        )

        st.download_button(
            "Download CSV",
            filtered_df[view_columns].to_csv(index=False).encode("utf-8"),
            file_name="campaign_health_check.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")
