import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import ast
import json

st.set_page_config(page_title="Uber vs Ola — Review Analytics", layout="wide")


# =========================
# 1) 数据加载与健壮清洗
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # 1) 读取
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # 2) 兼容列名（可按你实际情况继续补）
    rename_map = {
        "thumbs": "thumbs_up",
        "likes": "thumbs_up",
        "developer_replied": "has_response",
        "resp_delay": "resp_delay_days",
        "topic_list": "neg_topics",
        "neg_topic_list": "neg_topics",
        "createdAt": "created_at",
        "created_time": "created_at",
        "date": "created_at",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3) 解析 neg_topics 为 list（极其常见：CSV 会把 list 变字符串）
    def parse_topics(x):
        # None
        if x is None:
            return []
        # NaN（只对标量判断）
        if isinstance(x, float) and np.isnan(x):
            return []

        # 已经是 list
        if isinstance(x, list):
            return x

        # numpy array / tuple / set -> list
        if isinstance(x, (np.ndarray, tuple, set)):
            return list(x)

        # 字符串：尝试 JSON 或 Python literal
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in {"nan", "none", "null"}:
                return []
            try:
                v = json.loads(s)  # '["gps","pricing"]'
                return v if isinstance(v, list) else []
            except Exception:
                pass
            try:
                v = ast.literal_eval(s)  # "['gps','pricing']"
                return v if isinstance(v, list) else []
            except Exception:
                return []

        # 兜底
        return []

    if "neg_topics" in df.columns:
        df["neg_topics"] = df["neg_topics"].apply(parse_topics)

    # 4) 推导 is_problem / is_low_rating（如果缺失）
    if "is_problem" not in df.columns:
        if "k_neg" in df.columns:
            df["is_problem"] = (df["k_neg"].fillna(0) > 0).astype(int)
        elif "neg_topics" in df.columns:
            df["is_problem"] = df["neg_topics"].apply(lambda x: int(isinstance(x, list) and len(x) > 0))
        else:
            df["is_problem"] = 0

    if "is_low_rating" not in df.columns:
        if "rating" in df.columns:
            df["is_low_rating"] = (df["rating"] <= 2).astype(int)
        else:
            df["is_low_rating"] = 0

    # 5) 生成 month_dt：优先用 month，其次尝试 created_at
    if "month" in df.columns:
        # 可能是 period / string / datetime
        if pd.api.types.is_period_dtype(df["month"]):
            df["month_dt"] = df["month"].dt.to_timestamp()
        else:
            df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
    else:
        # 尝试用 created_at 生成 month_dt
        if "created_at" in df.columns:
            dt = pd.to_datetime(df["created_at"], errors="coerce")
            df["month_dt"] = dt.dt.to_period("M").dt.to_timestamp()
        else:
            # 没有任何时间字段，给一个空列（后续会提示）
            df["month_dt"] = pd.NaT

    # 6) 一些列缺失的兜底（防止 groupby 聚合时报错）
    for col, default in [
        ("thumbs_up", np.nan),
        ("has_response", 0),
        ("resp_delay_days", np.nan),
        ("k_neg", np.nan),
        ("is_multi2", 0),
        ("review_id", None),
        ("service", None),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


# =========================
# 2) 指标计算
# =========================
@st.cache_data(show_spinner=False)
def compute_metrics(df_all: pd.DataFrame):
    # 如果 month_dt 没有有效值，就直接返回空，避免崩
    if "month_dt" not in df_all.columns or df_all["month_dt"].notna().sum() == 0:
        monthly = pd.DataFrame()
    else:
        monthly = df_all.groupby(["service", "month_dt"]).agg(
            n=("review_id", "count"),
            mean_rating=("rating", "mean"),
            low_rating_rate=("is_low_rating", "mean"),
            avg_complexity=("k_neg", "mean"),
            multi2_rate=("is_multi2", "mean"),
            problem_rate=("is_problem", "mean"),
            response_rate=("has_response", "mean"),
        ).reset_index().rename(columns={"month_dt": "month"})

    resp_summary = df_all.groupby("service").agg(
        response_rate=("has_response", "mean"),
        median_delay=("resp_delay_days", "median"),
        p95_delay=("resp_delay_days", lambda s: np.nanpercentile(s.dropna(), 95) if s.dropna().shape[0] > 0 else np.nan),
        n=("review_id", "count")
    ).reset_index()

    d = df_all[df_all["is_problem"] == 1].copy()
    bias = d.groupby(["service", "has_response"]).agg(
        n=("review_id", "count"),
        mean_rating=("rating", "mean"),
        low_rating_rate=("is_low_rating", "mean"),
        mean_thumbs=("thumbs_up", "mean"),
        avg_complexity=("k_neg", "mean"),
    ).reset_index()

    # topic_profile（依赖 neg_topics + is_problem）
    df_prob = df_all[df_all["is_problem"] == 1].copy()

    all_topics = set()
    if "neg_topics" in df_prob.columns and len(df_prob) > 0:
        for lst in df_prob["neg_topics"]:
            if isinstance(lst, list) and len(lst) > 0:
                all_topics.update(lst)
    topics = sorted(all_topics)

    def build_topic_profile(df_prob_svc, topics):
        rows = []
        if "neg_topics" not in df_prob_svc.columns or len(df_prob_svc) == 0:
            return pd.DataFrame()

        for t in topics:
            mask = df_prob_svc["neg_topics"].apply(lambda lst: isinstance(lst, list) and (t in lst))
            n = int(mask.sum())
            if n == 0:
                continue
            sub = df_prob_svc.loc[mask]
            rows.append({
                "topic": t,
                "n_reviews": n,
                "share_in_problem": n / (len(df_prob_svc) + 1e-9),
                "mean_rating": sub["rating"].mean(),
                "low_rating_rate": (sub["rating"] <= 2).mean() if "rating" in sub.columns else np.nan,
                "mean_thumbs": sub["thumbs_up"].mean(),
                "p_has_response": sub["has_response"].mean(),
                "median_resp_days": np.nanmedian(sub.loc[sub["has_response"] == 1, "resp_delay_days"].values),
                "avg_complexity": sub["k_neg"].mean(),
                "service": sub["service"].iloc[0] if "service" in sub.columns and len(sub) > 0 else None
            })
        return pd.DataFrame(rows)

    topic_profiles = []
    for svc in df_prob["service"].dropna().unique():
        tp = build_topic_profile(df_prob[df_prob["service"] == svc], topics)
        if len(tp) > 0:
            topic_profiles.append(tp)
    topic_profile = pd.concat(topic_profiles, ignore_index=True) if topic_profiles else pd.DataFrame()

    # rank / ops_priority_score
    rank = topic_profile.copy()
    if len(rank) > 0:
        rank["thumbs_log"] = np.log1p(rank["mean_thumbs"].fillna(0))
        rank["uncovered"] = 1 - rank["p_has_response"].fillna(0)

        denom = rank["thumbs_log"].max() + 1e-9
        rank["ops_priority_score"] = (
            1.2 * rank["share_in_problem"].fillna(0)
            + 1.8 * rank["low_rating_rate"].fillna(0)
            + 0.7 * (rank["thumbs_log"] / denom)
            + 0.6 * rank["uncovered"]
        )
        rank = rank.sort_values(["service", "ops_priority_score"], ascending=[True, False])

    # pivot delta
    if len(topic_profile) > 0:
        pivot = topic_profile.pivot_table(
            index="topic", columns="service",
            values=["share_in_problem", "low_rating_rate", "mean_thumbs"],
            aggfunc="mean"
        )
        pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()

        if "low_rating_rate_Uber" in pivot.columns and "low_rating_rate_Ola" in pivot.columns:
            pivot["delta_low_rating_rate"] = pivot["low_rating_rate_Uber"] - pivot["low_rating_rate_Ola"]
            pivot["delta_share"] = pivot["share_in_problem_Uber"] - pivot["share_in_problem_Ola"]
            pivot["delta_thumbs"] = pivot["mean_thumbs_Uber"] - pivot["mean_thumbs_Ola"]
    else:
        pivot = pd.DataFrame()

    return monthly, resp_summary, bias, topic_profile, rank, pivot


# =========================
# 3) Sidebar
# =========================
st.sidebar.title("Filters")
data_path = st.sidebar.text_input("Data path", value="df_all.parquet")

try:
    df_all = load_data(data_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

monthly, resp_summary, bias, topic_profile, rank, pivot = compute_metrics(df_all)

# Debug 小面板（必要时打开）
with st.sidebar.expander("Debug", expanded=False):
    st.write("Columns:", df_all.columns.tolist())
    if "neg_topics" in df_all.columns:
        st.write("neg_topics types (head):", df_all["neg_topics"].head(5).apply(type).tolist())
        st.write(df_all["neg_topics"].head(5))
    st.write("month_dt valid:", int(df_all["month_dt"].notna().sum()))

services = sorted([s for s in df_all["service"].dropna().unique().tolist() if s is not None])
svc_sel = st.sidebar.multiselect("Service", services, default=services)

# 月份范围：若 month_dt 无效则给一个兜底
if df_all["month_dt"].notna().sum() > 0:
    min_month = df_all["month_dt"].min()
    max_month = df_all["month_dt"].max()
    month_range = st.sidebar.slider(
        "Month range",
        min_value=min_month.to_pydatetime(),
        max_value=max_month.to_pydatetime(),
        value=(min_month.to_pydatetime(), max_month.to_pydatetime())
    )
else:
    month_range = None
    st.sidebar.warning("No valid month_dt found. Trend charts will be empty.")

top_n = st.sidebar.slider("Top N topics", 5, 20, 12)

# 应用筛选
df_f = df_all[df_all["service"].isin(svc_sel)].copy()
if month_range is not None:
    df_f = df_f[
        (df_f["month_dt"] >= pd.to_datetime(month_range[0])) &
        (df_f["month_dt"] <= pd.to_datetime(month_range[1]))
    ].copy()

monthly_f, resp_summary_f, bias_f, topic_profile_f, rank_f, pivot_f = compute_metrics(df_f)


# =========================
# 4) Main UI
# =========================
st.title("Uber vs Ola — Review Analytics Dashboard")
tab_overview, tab_topics, tab_response, tab_anom = st.tabs(["Overview", "Topics", "Response", "Anomaly"])


# ---------- Overview ----------
with tab_overview:
    st.subheader("Key Metrics")

    kpi = df_f.groupby("service").agg(
        reviews=("review_id", "count"),
        mean_rating=("rating", "mean"),
        low_rating_rate=("is_low_rating", "mean"),
        problem_rate=("is_problem", "mean"),
        response_rate=("has_response", "mean")
    ).reset_index()

    c1, c2, c3, c4, c5 = st.columns(5)
    if len(kpi) > 0:
        c1.metric("Reviews", f"{int(kpi['reviews'].sum()):,}")
        c2.metric("Mean rating", f"{kpi['mean_rating'].mean():.2f}")
        c3.metric("Low-rating rate", f"{kpi['low_rating_rate'].mean()*100:.1f}%")
        c4.metric("Problem rate", f"{kpi['problem_rate'].mean()*100:.1f}%")
        c5.metric("Response rate", f"{kpi['response_rate'].mean()*100:.1f}%")

    st.divider()
    st.subheader("Monthly Trends")

    colA, colB = st.columns(2)

    with colA:
        if len(monthly_f) == 0:
            st.info("No monthly trend data (month_dt missing or invalid).")
        else:
            fig1 = px.line(monthly_f, x="month", y="avg_complexity", color="service",
                           markers=True, title="Monthly Failure Complexity (avg #negative topics)")
            fig1.update_layout(legend_title_text="", height=360)
            st.plotly_chart(fig1, use_container_width=True)

    with colB:
        if len(monthly_f) == 0:
            st.info("No monthly trend data (month_dt missing or invalid).")
        else:
            fig2 = px.line(monthly_f, x="month", y="mean_rating", color="service",
                           markers=True, title="Monthly Mean Rating")
            fig2.update_layout(legend_title_text="", height=360)
            st.plotly_chart(fig2, use_container_width=True)


# ---------- Topics ----------
with tab_topics:
    st.subheader("Operational Priority Topics")

    if len(rank_f) == 0:
        st.info("No topic data found. Please check: `neg_topics` parsed into list AND `is_problem` is available.")
        st.write("Tip: open sidebar Debug to see neg_topics sample and types.")
    else:
        for svc in svc_sel:
            sub = rank_f[rank_f["service"] == svc].head(top_n).copy()
            # 为了“最大在上”，用 autorange reversed + 升序绘制
            sub = sub.sort_values("ops_priority_score", ascending=True)

            fig = px.bar(
                sub,
                x="ops_priority_score",
                y="topic",
                orientation="h",
                title=f"{svc} — Operational Priority (Top {top_n})",
                text=sub["ops_priority_score"].round(2)
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=420,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Ops priority score",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Topic Delta (Uber − Ola)")

        if len(pivot_f) == 0 or ("delta_low_rating_rate" not in pivot_f.columns):
            st.info("Delta view requires both Uber and Ola in the filtered data.")
        else:
            metric = st.selectbox("Delta metric", ["delta_low_rating_rate", "delta_share", "delta_thumbs"])
            tmp = pivot_f.dropna(subset=[metric]).copy()
            tmp["abs"] = tmp[metric].abs()
            tmp = tmp.sort_values("abs", ascending=False).head(top_n).sort_values(metric, ascending=True)

            figd = px.bar(
                tmp, x=metric, y="topic", orientation="h",
                title=f"Top {top_n} topic deltas (Uber − Ola): {metric}",
                text=tmp[metric].round(3)
            )
            figd.update_layout(
                height=420,
                yaxis_title="",
                xaxis_title=metric,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(figd, use_container_width=True)

        st.divider()
        st.subheader("Priority Table")
        show_cols = ["service", "topic", "ops_priority_score", "share_in_problem", "low_rating_rate",
                     "mean_thumbs", "p_has_response", "median_resp_days"]
        st.dataframe(rank_f[show_cols].sort_values(["service", "ops_priority_score"], ascending=[True, False]).head(200))


# ---------- Response ----------
with tab_response:
    st.subheader("Response Summary")
    st.dataframe(resp_summary_f)

    st.divider()
    st.subheader("CDF of Developer Response Delay (log scale)")

    df_resp = df_f[df_f["has_response"] == 1].copy()
    df_resp = df_resp[df_resp["resp_delay_days"].notna()]
    df_resp = df_resp[df_resp["resp_delay_days"] >= 0]

    if len(df_resp) == 0:
        st.info("No responded reviews in the selected range.")
    else:
        rows = []
        for svc in svc_sel:
            s = df_resp[df_resp["service"] == svc]["resp_delay_days"].values
            if len(s) == 0:
                continue
            s = np.sort(s)
            y = np.arange(1, len(s) + 1) / len(s)
            rows.append(pd.DataFrame({"service": svc, "delay": s, "cdf": y}))
        cdf_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        if len(cdf_df) == 0:
            st.info("No response delay data for selected services.")
        else:
            figcdf = px.line(cdf_df, x="delay", y="cdf", color="service", title="CDF of response delay")
            figcdf.update_xaxes(type="log", title="Response delay (days, log scale)")
            figcdf.update_yaxes(title="Cumulative proportion")
            figcdf.update_layout(height=420, legend_title_text="")
            st.plotly_chart(figcdf, use_container_width=True)

    st.divider()
    st.subheader("Are responded problem reviews more viral? (thumbs up)")
    figb = px.bar(
        bias_f, x="has_response", y="mean_thumbs", color="service",
        barmode="group", title="Mean thumbs_up by response status (problem reviews)"
    )
    figb.update_layout(height=380, legend_title_text="", xaxis_title="Has response", yaxis_title="Mean thumbs_up")
    st.plotly_chart(figb, use_container_width=True)


# ---------- Anomaly ----------
with tab_anom:
    st.subheader("Rolling z-score anomalies on avg_complexity")

    if len(monthly_f) == 0:
        st.info("No monthly data available for anomaly detection.")
    else:
        def add_rolling_z(df_m, value_col="avg_complexity", window=6):
            df_m = df_m.sort_values("month").copy()
            v = df_m[value_col]
            mu = v.rolling(window, min_periods=3).mean()
            sd = v.rolling(window, min_periods=3).std()
            df_m["z"] = (v - mu) / (sd + 1e-9)
            return df_m

        Z_THR = st.slider("Z threshold", 1.0, 4.0, 2.0, 0.1)

        zz = []
        for svc in svc_sel:
            d = monthly_f[monthly_f["service"] == svc].copy()
            d = add_rolling_z(d, value_col="avg_complexity", window=6)
            zz.append(d)
        monthly_z = pd.concat(zz, ignore_index=True) if zz else pd.DataFrame()

        figz = px.line(monthly_z, x="month", y="z", color="service", markers=True,
                       title="Anomaly signal: rolling z-score of avg_complexity")
        figz.add_hline(y=Z_THR, line_dash="dash", opacity=0.4)
        figz.update_layout(height=420, legend_title_text="")
        st.plotly_chart(figz, use_container_width=True)

        st.subheader("Anomalous months (z > threshold)")
        out = monthly_z[monthly_z["z"] > Z_THR][["service", "month", "z"]].copy()
        out["month"] = out["month"].dt.strftime("%Y-%m")
        out = out.sort_values(["service", "z"], ascending=[True, False])
        st.dataframe(out)
