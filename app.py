import re
import pandas as pd
import streamlit as st

# ==============================
# 基础配置
# ==============================
st.set_page_config(page_title="Ride-hailing Review Insights Dashboard", layout="wide")

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
        raise ValueError(f"缺少必要列: {missing}. 你当前列为: {df.columns.tolist()}")

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
st.title("Ride-hailing Reviews — Dimension × Time Insights (AI Summary & Recommendations)")

# ==============================
# 侧边栏：选择查看模式
# ==============================
st.sidebar.header("查看设置")

view_mode_global = st.sidebar.radio(
    "查看模式",
    [
        "维度视角（单数据源）",
        "时间视角（单数据源）",
        "对比视角（Uber vs Ola）",  # ✅ 新增
    ],
    index=0
)

# 全局关键词搜索（所有模式都适用）
keyword = st.sidebar.text_input("关键词搜索（summary/reco/evidence）", value="").strip()

# 是否展示 Evidence（证据）
show_evidence = st.sidebar.checkbox("显示 Evidence（证据评论）", value=True)

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
                st.info("该时间段该维度无数据")
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
                st.info("该时间段该维度无数据")
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
if view_mode_global in ["维度视角（单数据源）", "时间视角（单数据源）"]:
    st.sidebar.header("数据源选择（单数据源模式）")

    source_name = st.sidebar.radio(
        "选择数据集",
        options=list(DATA_SOURCES.keys()),
        index=0,
        horizontal=False
    )

    excel_path = DATA_SOURCES[source_name]["excel_path"]
    sheet_name = DATA_SOURCES[source_name]["sheet_name"]

    try:
        df = load_data(excel_path, sheet_name)
    except Exception as e:
        st.error(f"读取 {source_name} 数据失败：{e}")
        st.stop()

    st.caption(f"当前数据源：**{source_name}**")

    if df["period_sort"].isna().any():
        st.warning("有部分 period 无法解析为时间（period_sort=NaT），排序可能不准确。请检查 period 格式（YYYYMM 或 YYYYQn）。")

    # ==============================
    # 模式 A：维度视角（单数据源）
    # ==============================
    if view_mode_global.startswith("维度视角"):
        st.sidebar.header("维度视角筛选")

        all_dims = sorted(df["dimension"].dropna().unique().tolist())
        dim_selected = st.sidebar.selectbox("选择维度 (dimension)", options=["(All)"] + all_dims, index=0)

        # 时间范围（按 period_sort）
        df_valid_time = df.dropna(subset=["period_sort"]).copy()
        if not df_valid_time.empty:
            min_t = df_valid_time["period_sort"].min()
            max_t = df_valid_time["period_sort"].max()
            time_range = st.sidebar.slider(
                "选择时间范围（按 period 排序）",
                min_value=min_t.to_pydatetime(),
                max_value=max_t.to_pydatetime(),
                value=(min_t.to_pydatetime(), max_t.to_pydatetime()),
                format="YYYY-MM"
            )
        else:
            time_range = None
            st.sidebar.info("没有可解析的 period_sort，时间滑块不可用。")

        view_mode = st.sidebar.radio("展示形式", ["卡片（按时间排序）", "表格"], index=0)
        page_size = st.sidebar.selectbox("卡片：每页条数", [10, 20, 50, 100], index=1)

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
        c1.metric("筛选后条目数", len(dff))
        c2.metric("维度数", dff["dimension"].nunique())
        c3.metric("时间段数", dff["period"].nunique())
        st.divider()

        if dff.empty:
            st.info("筛选结果为空。请调整维度/时间范围/关键词。")
            st.stop()

        if view_mode.startswith("表格"):
            show_cols = ["dimension", "period", "ai_summary", "ai_recommendations"]
            if "negative_reviews_concat" in dff.columns:
                show_cols.append("negative_reviews_concat")
            st.dataframe(dff[show_cols], use_container_width=True, height=700)

        else:
            total = len(dff)
            page_count = (total + page_size - 1) // page_size
            page = st.number_input("页码", min_value=1, max_value=page_count, value=1, step=1)

            start = (page - 1) * page_size
            end = min(start + page_size, total)
            st.caption(f"显示第 {start+1} - {end} 条 / 共 {total} 条")

            dff_page = dff.iloc[start:end]
            for _, row in dff_page.iterrows():
                render_single_card(row)

    # ==============================
    # 模式 B：时间视角（单数据源）
    # ==============================
    else:
        st.sidebar.header("时间视角筛选")

        tmp = df.dropna(subset=["period"]).copy()
        tmp["period_sort2"] = tmp["period"].apply(period_to_sortkey)

        periods_df = (tmp[["period", "period_sort2"]]
                      .drop_duplicates()
                      .sort_values(["period_sort2", "period"], ascending=[True, True]))

        period_list = periods_df["period"].tolist()
        if not period_list:
            st.error("数据中没有可用的 period。")
            st.stop()

        period_selected = st.sidebar.selectbox("选择时间 (period)", options=period_list, index=len(period_list) - 1)

        all_dims = sorted(df["dimension"].dropna().unique().tolist())
        dims_selected = st.sidebar.multiselect("选择维度（默认全选）", options=all_dims, default=all_dims)

        view_mode = st.sidebar.radio("展示形式", ["网格卡片（推荐）", "表格"], index=0)

        dff = df[df["period"] == period_selected].copy()

        if dims_selected:
            dff = dff[dff["dimension"].isin(dims_selected)]

        dff = apply_keyword_filter(dff, keyword)
        dff = dff.sort_values(["dimension"], ascending=[True])

        # 顶部概览
        c1, c2, c3 = st.columns(3)
        c1.metric("当前 period", period_selected)
        c2.metric("展示维度数", dff["dimension"].nunique())
        c3.metric("条目数", len(dff))
        st.divider()

        if dff.empty:
            st.info("该时间点筛选结果为空（可能该 period 没有某些维度，或关键词过滤后为空）。")
            st.stop()

        if view_mode.startswith("表格"):
            show_cols = ["dimension", "period", "ai_summary", "ai_recommendations"]
            if "negative_reviews_concat" in dff.columns:
                show_cols.append("negative_reviews_concat")
            st.dataframe(dff[show_cols], use_container_width=True, height=700)

        else:
            cols_per_row = st.sidebar.selectbox("网格：每行卡片列数", [2, 3], index=0)
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
    st.sidebar.header("对比视角设置（Uber vs Ola）")

    # 读取两份数据
    try:
        df_uber = load_data(DATA_SOURCES["Uber"]["excel_path"], DATA_SOURCES["Uber"]["sheet_name"])
    except Exception as e:
        st.error(f"读取 Uber 数据失败：{e}")
        st.stop()

    try:
        df_ola = load_data(DATA_SOURCES["Ola"]["excel_path"], DATA_SOURCES["Ola"]["sheet_name"])
    except Exception as e:
        st.error(f"读取 Ola 数据失败：{e}")
        st.stop()

    # period 列表：取两者 union，并按 sortkey 排序
    p1 = df_uber[["period"]].dropna().drop_duplicates()
    p2 = df_ola[["period"]].dropna().drop_duplicates()
    periods = pd.concat([p1, p2], ignore_index=True).drop_duplicates()
    periods["period_sort"] = periods["period"].apply(period_to_sortkey)
    periods = periods.sort_values(["period_sort", "period"], ascending=[True, True])
    period_list = periods["period"].tolist()

    if not period_list:
        st.error("Uber/Ola 数据中都没有可用的 period。")
        st.stop()

    # 选择一个时间点
    period_selected = st.sidebar.selectbox("选择时间 (period)", options=period_list, index=len(period_list) - 1)

    # 维度列表：取两者 union
    d1 = df_uber[["dimension"]].dropna().drop_duplicates()
    d2 = df_ola[["dimension"]].dropna().drop_duplicates()
    dims_all = pd.concat([d1, d2], ignore_index=True).drop_duplicates()["dimension"].tolist()
    dims_all = sorted(dims_all)

    dims_selected = st.sidebar.multiselect("选择维度（默认全选）", options=dims_all, default=dims_all)

    compare_layout = st.sidebar.radio("对比展示形式", ["按维度卡片（并排对比）", "表格（对比列）"], index=0)

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
    c1.metric("当前 period", period_selected)
    c2.metric("展示维度数", len(dims_show))
    c3.metric("关键词过滤", "ON" if keyword else "OFF")
    st.divider()

    if not dims_show:
        st.info("该时间点在关键词过滤后没有可展示的维度。请调整关键词或时间。")
        st.stop()

    if compare_layout.startswith("表格"):
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
                row["uber_evidence"] = (u.get("negative_reviews_concat", "") if u is not None and "negative_reviews_concat" in u.index else "")
                row["ola_evidence"] = (o.get("negative_reviews_concat", "") if o is not None and "negative_reviews_concat" in o.index else "")
            rows.append(row)

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True, height=750)

    else:
        # 卡片对比：每个维度一张并排卡片
        cols_per_row = st.sidebar.selectbox("卡片：每行维度数", [1, 2], index=0)

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
