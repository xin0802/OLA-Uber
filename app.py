import re
import pandas as pd
import streamlit as st

# ==============================
# 基础配置
# ==============================
st.set_page_config(page_title="Review Insights Dashboard", layout="wide")

# ==============================
# 1) 数据源配置：把路径改成你的真实文件名/相对路径
# ==============================
DATA_SOURCES = {
    "Uber": {
        "excel_path": "dimension_period_negative_reviews_with_ai_summary_reco.xlsx",
        "sheet_name": "summary",
    },
    "Ola": {
        "excel_path": "ola_dimension_period_negative_reviews_with_ai_summary_reco.xlsx",  # ← 改成你的 Ola 文件名/路径
        "sheet_name": "summary",
    },
}

# ==============================
# 工具函数：把 period 字符串转成可排序的“时间索引”
# 支持两种格式：
# - 月：YYYYMM（例如 201712）
# - 季度：YYYYQn（例如 2017Q3）
# ==============================
def period_to_sortkey(p: str):
    p = str(p).strip()

    # 月：201712
    if re.fullmatch(r"\d{6}", p):
        return pd.Period(p, freq="M").to_timestamp()

    # 季度：2017Q3
    m = re.fullmatch(r"(\d{4})Q([1-4])", p)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = 1 + (q - 1) * 3
        return pd.Timestamp(year=year, month=month, day=1)

    return pd.NaT

# ==============================
# 读取数据（用缓存加速）
# 注意：缓存函数参数不同会触发不同缓存（即 Uber/Ola 分开缓存）
# ==============================
@st.cache_data(show_spinner=False)
def load_data(path: str, sheet: str):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]  # 清理列名空格

    # 必要列检查
    required = {"dimension", "period", "ai_summary", "ai_recommendations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Current columns: {df.columns.tolist()}")

    # 生成排序用时间 key
    df["period_sort"] = df["period"].apply(period_to_sortkey)

    # 统一空值（展示更干净）
    for c in ["ai_summary", "ai_recommendations"]:
        df[c] = df[c].fillna("").astype(str)

    if "negative_reviews_concat" in df.columns:
        df["negative_reviews_concat"] = df["negative_reviews_concat"].fillna("").astype(str)

    return df

# ==============================
# 页面标题
# ==============================
st.title("Negative Reviews Analysis Dashboard")

# ==============================
# 侧边栏：选择查看模式
# ==============================
st.sidebar.header("View Settings")

view_mode_global = st.sidebar.radio(
    "View Mode",
    [
        "Dimension View (Single Source)",
        "Time View (Single Source)",
        "Comparison View (Uber vs Ola)",  # ✅ 新增
    ],
    index=0
)

# 全局关键词搜索（所有模式都适用）
keyword = st.sidebar.text_input("Keyword Search (summary / recommendations / evidence)", value="").strip()

# 是否展示 Evidence（证据）
show_evidence = st.sidebar.checkbox("Show Evidence (negative reviews)", value=True)

# ==============================
# 工具：关键词筛选
# ==============================
def apply_keyword_filter(dff: pd.DataFrame, kw: str) -> pd.DataFrame:
    if not kw:
        return dff

    cols_for_search = ["ai_summary", "ai_recommendations"]
    if "negative_reviews_concat" in dff.columns:
        cols_for_search.append("negative_reviews_concat")

    search_text = dff[cols_for_search].fillna("").astype(str).agg("\n".join, axis=1)
    return dff[search_text.str.contains(kw, case=False, na=False)]

# ==============================
# 工具：渲染单数据源卡片
# ==============================
def render_single_card(row: pd.Series):
    dimension = row.get("dimension", "")
    period = row.get("period", "")
    summary = row.get("ai_summary", "")
    reco = row.get("ai_recommendations", "")
    evidence = row.get("negative_reviews_concat", "")

    with st.container(border=True):
        st.subheader(f"{dimension}  |  {period}")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**AI Summary**")
            st.write(summary if summary.strip() else "(empty)")
        with colB:
            st.markdown("**AI Recommendations**")
            st.write(reco if reco.strip() else "(empty)")

        if show_evidence and "negative_reviews_concat" in row.index:
            if str(evidence).strip():
                with st.expander("Evidence (Top negative reviews, concatenated)"):
                    st.text(evidence)
            else:
                with st.expander("Evidence (empty)"):
                    st.text("(empty)")

# ==============================
# 工具：渲染对比卡片（同一维度 & 同一时间：Uber vs Ola 并排）
# ==============================
def render_compare_card(dimension: str, period: str, uber_row: pd.Series | None, ola_row: pd.Series | None):
    with st.container(border=True):
        st.subheader(f"{dimension}  |  {period}")

        left, right = st.columns(2)

        # ---------- Uber ----------
        with left:
            st.markdown("### Uber")
            if uber_row is None:
                st.info("No data for this dimension in the selected period.")
            else:
                st.markdown("**AI Summary**")
                st.write(uber_row.get("ai_summary", "").strip() or "(empty)")
                st.markdown("**AI Recommendations**")
                st.write(uber_row.get("ai_recommendations", "").strip() or "(empty)")

                if show_evidence and "negative_reviews_concat" in uber_row.index:
                    ev = uber_row.get("negative_reviews_concat", "")
                    with st.expander("Evidence (Uber)"):
                        st.text(ev.strip() or "(empty)")

        # ---------- Ola ----------
        with right:
            st.markdown("### Ola")
            if ola_row is None:
                st.info("No data for this dimension in the selected period.")
            else:
                st.markdown("**AI Summary**")
                st.write(ola_row.get("ai_summary", "").strip() or "(empty)")
                st.markdown("**AI Recommendations**")
                st.write(ola_row.get("ai_recommendations", "").strip() or "(empty)")

                if show_evidence and "negative_reviews_concat" in ola_row.index:
                    ev = ola_row.get("negative_reviews_concat", "")
                    with st.expander("Evidence (Ola)"):
                        st.text(ev.strip() or "(empty)")

# ==============================
# 模式 A/B：单数据源模式，需要先选数据源并加载 df
# ==============================
if view_mode_global in ["Dimension View (Single Source)", "Time View (Single Source)"]:
    st.sidebar.header("Data Source (Single Source Mode)")

    source_name = st.sidebar.radio(
        "Select Dataset",
        options=list(DATA_SOURCES.keys()),
        index=0,
        horizontal=False
    )

    excel_path = DATA_SOURCES[source_name]["excel_path"]
    sheet_name = DATA_SOURCES[source_name]["sheet_name"]

    try:
        df = load_data(excel_path, sheet_name)
    except Exception as e:
        st.error(f"Failed to load {source_name} data: {e}")
        st.stop()

    st.caption(f"Current source: **{source_name}**")

    if df["period_sort"].isna().any():
        st.warning(
            "Some 'period' values could not be parsed into timestamps (period_sort=NaT). "
            "Sorting may be inaccurate. Please ensure 'period' is in YYYYMM or YYYYQn format."
        )

    # ==============================
    # 模式 A：维度视角（单数据源）
    # ==============================
    if view_mode_global.startswith("Dimension View"):
        st.sidebar.header("Filters (Dimension View)")

        all_dims = sorted(df["dimension"].dropna().unique().tolist())
        dim_selected = st.sidebar.selectbox("Select Dimension", options=["(All)"] + all_dims, index=0)

        # 时间范围（按 period_sort）
        df_valid_time = df.dropna(subset=["period_sort"]).copy()
        if not df_valid_time.empty:
            min_t = df_valid_time["period_sort"].min()
            max_t = df_valid_time["period_sort"].max()
            time_range = st.sidebar.slider(
                "Select Time Range (based on parsed period)",
                min_value=min_t.to_pydatetime(),
                max_value=max_t.to_pydatetime(),
                value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
                format="YYYY-MM"
            )
        else:
            time_range = None
            st.sidebar.info("No parsable period_sort found. Time range slider is disabled.")

        view_mode = st.sidebar.radio("Display Format", ["Cards (sorted by time)", "Table"], index=0)
        page_size = st.sidebar.selectbox("Cards: items per page", [10, 20, 50, 100], index=1)

        dff = df.copy()

        if dim_selected != "(All)":
            dff = dff[dff["dimension"] == dim_selected]

        if time_range is not None and not df_valid_time.empty:
            start_dt, end_dt = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
            dff = dff.dropna(subset=["period_sort"])
            dff = dff[(dff["period_sort"] >= start_dt) & (dff["period_sort"] <= end_dt)]

        dff = apply_keyword_filter(dff, keyword)

        dff = dff.sort_values(["dimension", "period_sort", "period"], ascending=[True, True, True])

        # 顶部概览
        c1, c2, c3 = st.columns(3)
        c1.metric("Records (after filtering)", len(dff))
        c2.metric("No. of Dimensions", dff["dimension"].nunique())
        c3.metric("No. of Periods", dff["period"].nunique())
        st.divider()

        if dff.empty:
            st.info("No results. Please adjust filters (dimension / time range / keyword).")
            st.stop()

        if view_mode.startswith("Table"):
            show_cols = ["dimension", "period", "ai_summary", "ai_recommendations"]
            if "negative_reviews_concat" in dff.columns:
                show_cols.append("negative_reviews_concat")
            st.dataframe(dff[show_cols], use_container_width=True, height=700)

        else:
            total = len(dff)
            page_count = (total + page_size - 1) // page_size
            page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)

            start = (page - 1) * page_size
            end = min(start + page_size, total)
            st.caption(f"Showing {start+1}–{end} of {total}")

            dff_page = dff.iloc[start:end]
            for _, row in dff_page.iterrows():
                render_single_card(row)

    # ==============================
    # 模式 B：时间视角（单数据源）
    # ==============================
    else:
        st.sidebar.header("Filters (Time View)")

        tmp = df.dropna(subset=["period"]).copy()
        tmp["period_sort2"] = tmp["period"].apply(period_to_sortkey)

        periods_df = (
            tmp[["period", "period_sort2"]]
            .drop_duplicates()
            .sort_values(["period_sort2", "period"], ascending=[True, True])
        )

        period_list = periods_df["period"].tolist()
        if not period_list:
            st.error("No usable 'period' values found in the dataset.")
            st.stop()

        period_selected = st.sidebar.selectbox("Select Period", options=period_list, index=len(period_list) - 1)

        all_dims = sorted(df["dimension"].dropna().unique().tolist())
        dims_selected = st.sidebar.multiselect("Select Dimensions (default: all)", options=all_dims, default=all_dims)

        view_mode = st.sidebar.radio("Display Format", ["Grid Cards (recommended)", "Table"], index=0)

        dff = df[df["period"] == period_selected].copy()

        if dims_selected:
            dff = dff[dff["dimension"].isin(dims_selected)]

        dff = apply_keyword_filter(dff, keyword)
        dff = dff.sort_values(["dimension"], ascending=[True])

        # 顶部概览
        c1, c2, c3 = st.columns(3)
        c1.metric("Current period", period_selected)
        c2.metric("Dimensions shown", dff["dimension"].nunique())
        c3.metric("Records", len(dff))
        st.divider()

        if dff.empty:
            st.info("No results for this period after filtering (dimensions/keyword).")
            st.stop()

        if view_mode.startswith("Table"):
            show_cols = ["dimension", "period", "ai_summary", "ai_recommendations"]
            if "negative_reviews_concat" in dff.columns:
                show_cols.append("negative_reviews_concat")
            st.dataframe(dff[show_cols], use_container_width=True, height=700)

        else:
            cols_per_row = st.sidebar.selectbox("Grid: cards per row", [2, 3], index=0)
            rows = dff.to_dict(orient="records")

            for i in range(0, len(rows), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j >= len(rows):
                        break
                    with cols[j]:
                        render_single_card(pd.Series(rows[i + j]))

# ==============================
# 模式 C：对比视角（Uber vs Ola）
# ==============================
else:
    st.sidebar.header("Comparison Settings (Uber vs Ola)")

    # 读取两份数据
    try:
        df_uber = load_data(DATA_SOURCES["Uber"]["excel_path"], DATA_SOURCES["Uber"]["sheet_name"])
    except Exception as e:
        st.error(f"Failed to load Uber data: {e}")
        st.stop()

    try:
        df_ola = load_data(DATA_SOURCES["Ola"]["excel_path"], DATA_SOURCES["Ola"]["sheet_name"])
    except Exception as e:
        st.error(f"Failed to load Ola data: {e}")
        st.stop()

    # period 列表：取两者 union，并按 sortkey 排序
    p1 = df_uber[["period"]].dropna().drop_duplicates()
    p2 = df_ola[["period"]].dropna().drop_duplicates()
    periods = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
    periods["period_sort"] = periods["period"].apply(period_to_sortkey)
    periods = periods.sort_values(["period_sort", "period"], ascending=[True, True])
    period_list = periods["period"].tolist()

    if not period_list:
        st.error("No usable 'period' values found in Uber/Ola datasets.")
        st.stop()

    # 选择一个时间点
    period_selected = st.sidebar.selectbox("Select Period", options=period_list, index=len(period_list) - 1)

    # 维度列表：取两者 union
    d1 = df_uber[["dimension"]].dropna().drop_duplicates()
    d2 = df_ola[["dimension"]].dropna().drop_duplicates()
    dims_all = pd.concat([d1, d2], ignore_index=True).drop_duplicates()["dimension"].tolist()
    dims_all = sorted(dims_all)

    dims_selected = st.sidebar.multiselect("Select Dimensions (default: all)", options=dims_all, default=dims_all)

    compare_layout = st.sidebar.radio(
        "Comparison Display",
        ["Side-by-side Cards (by dimension)", "Table (side-by-side columns)"],
        index=0
    )

    # 过滤：固定 period
    uber_p = df_uber[df_uber["period"] == period_selected].copy()
    ola_p = df_ola[df_ola["period"] == period_selected].copy()

    # 过滤：维度集合
    if dims_selected:
        uber_p = uber_p[uber_p["dimension"].isin(dims_selected)]
        ola_p = ola_p[ola_p["dimension"].isin(dims_selected)]

    # 关键词筛选（分别在各自数据里筛，然后 union 维度）
    uber_p_f = apply_keyword_filter(uber_p, keyword)
    ola_p_f = apply_keyword_filter(ola_p, keyword)

    # 如果有关键词：只保留“Uber 命中 or Ola 命中”的维度
    if keyword:
        dims_hit = sorted(set(uber_p_f["dimension"].tolist()) | set(ola_p_f["dimension"].tolist()))
        uber_p = uber_p[uber_p["dimension"].isin(dims_hit)]
        ola_p = ola_p[ola_p["dimension"].isin(dims_hit)]

    # 建索引：dimension -> row（同一 period 内通常每维度只有一行）
    uber_map = {r["dimension"]: r for _, r in uber_p.iterrows()}
    ola_map = {r["dimension"]: r for _, r in ola_p.iterrows()}

    # 最终要展示的维度集合（union）
    dims_show = sorted(set(uber_map.keys()) | set(ola_map.keys()))

    # 顶部概览
    c1, c2, c3 = st.columns(3)
    c1.metric("Current period", period_selected)
    c2.metric("Dimensions shown", len(dims_show))
    c3.metric("Keyword filter", "ON" if keyword else "OFF")
    st.divider()

    if not dims_show:
        st.info("No dimensions to display after filtering. Try adjusting the keyword or period.")
        st.stop()

    if compare_layout.startswith("Table"):
        # 表格对比：一行一个维度，Uber/Ola 各两列（summary/reco），可选 evidence
        rows = []
        for d in dims_show:
            u = uber_map.get(d)
            o = ola_map.get(d)

            row = {
                "dimension": d,
                "period": period_selected,
                "uber_summary": (u.get("ai_summary", "") if u is not None else ""),
                "uber_recommendations": (u.get("ai_recommendations", "") if u is not None else ""),
                "ola_summary": (o.get("ai_summary", "") if o is not None else ""),
                "ola_recommendations": (o.get("ai_recommendations", "") if o is not None else ""),
            }
            if show_evidence:
                row["uber_evidence"] = (
                    u.get("negative_reviews_concat", "")
                    if u is not None and "negative_reviews_concat" in u.index
                    else ""
                )
                row["ola_evidence"] = (
                    o.get("negative_reviews_concat", "")
                    if o is not None and "negative_reviews_concat" in o.index
                    else ""
                )
            rows.append(row)

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True, height=750)

    else:
        # 卡片对比：每个维度一张并排卡片
        cols_per_row = st.sidebar.selectbox("Cards: dimensions per row", [1, 2], index=0)

        for i in range(0, len(dims_show), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j >= len(dims_show):
                    break
                d = dims_show[i + j]
                with cols[j]:
                    u = uber_map.get(d)
                    o = ola_map.get(d)
                    render_compare_card(dimension=d, period=period_selected, uber_row=u, ola_row=o)
