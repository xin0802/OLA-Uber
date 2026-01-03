import re
import pandas as pd
import streamlit as st

# ==============================
# 基础配置
# ==============================
st.set_page_config(page_title="Uber Review Insights Dashboard", layout="wide")

# 你导出的 Excel 路径（建议用相对路径；部署到云端更稳）
EXCEL_PATH = "dimension_period_negative_reviews_with_ai_summary_reco.xlsx"
SHEET_NAME = "summary"

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
        # 以季度第一天作为排序 key（也可以改成季度最后一天）
        month = 1 + (q - 1) * 3
        return pd.Timestamp(year=year, month=month, day=1)

    # 兜底：无法解析就返回 NaT
    return pd.NaT

# ==============================
# 读取数据（用缓存加速）
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

    # 统一把空值转成空字符串（展示更干净）
    for c in ["ai_summary", "ai_recommendations"]:
        df[c] = df[c].fillna("").astype(str)

    if "negative_reviews_concat" in df.columns:
        df["negative_reviews_concat"] = df["negative_reviews_concat"].fillna("").astype(str)

    return df

# ==============================
# 页面标题
# ==============================
st.title("Uber Reviews — Dimension × Time Insights (AI Summary & Recommendations)")

# ==============================
# 加载数据
# ==============================
try:
    df = load_data(EXCEL_PATH, SHEET_NAME)
except Exception as e:
    st.error(f"读取数据失败：{e}")
    st.stop()

if df["period_sort"].isna().any():
    st.warning("有部分 period 无法解析为时间（period_sort=NaT），排序可能不准确。请检查 period 格式（YYYYMM 或 YYYYQn）。")

# ==============================
# 侧边栏：全局控件
# ==============================
st.sidebar.header("查看设置")

# 新增：查看模式
view_mode_global = st.sidebar.radio(
    "查看模式",
    ["维度视角（选维度 + 时间范围）", "时间视角（选时间 + 展示全部维度）"],
    index=0
)

# 全局关键词搜索（两种模式都适用）
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
# 工具：渲染一个维度卡片
# ==============================
def render_card(row: pd.Series):
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
            if evidence.strip():
                with st.expander("Evidence (Top negative reviews, concatenated)"):
                    st.text(evidence)
            else:
                with st.expander("Evidence (empty)"):
                    st.text("(empty)")

# ==============================
# 模式 1：维度视角（你原来的逻辑 + 小优化）
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

    # 展示模式（卡片 or 表格）
    view_mode = st.sidebar.radio("展示形式", ["卡片（按时间排序）", "表格"], index=0)
    page_size = st.sidebar.selectbox("卡片：每页条数", [10, 20, 50, 100], index=1)

    dff = df.copy()

    if dim_selected != "(All)":
        dff = dff[dff["dimension"] == dim_selected]

    if time_range is not None and not df_valid_time.empty:
        start_dt, end_dt = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
        dff = dff.dropna(subset=["period_sort"])
        dff = dff[(dff["period_sort"] >= start_dt) & (dff["period_sort"] <= end_dt)]

    # 关键词筛选
    dff = apply_keyword_filter(dff, keyword)

    # 排序
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
            render_card(row)

# ==============================
# 模式 2：时间视角（新增）：选一个 period → 展示该 period 下所有维度
# ==============================
else:
    st.sidebar.header("时间视角筛选")

    # period 下拉：按 period_sort 排序
    tmp = df.copy()
    tmp = tmp.dropna(subset=["period"]).copy()
    tmp["period_sort2"] = tmp["period"].apply(period_to_sortkey)

    # 去重并排序
    periods_df = (tmp[["period", "period_sort2"]]
                  .drop_duplicates()
                  .sort_values(["period_sort2", "period"], ascending=[True, True]))

    period_list = periods_df["period"].tolist()
    if not period_list:
        st.error("数据中没有可用的 period。")
        st.stop()

    # 选择一个时间点
    period_selected = st.sidebar.selectbox("选择时间 (period)", options=period_list, index=len(period_list) - 1)

    # 可选：只看某些维度（默认全 12 个）
    all_dims = sorted(df["dimension"].dropna().unique().tolist())
    dims_selected = st.sidebar.multiselect("选择维度（默认全选）", options=all_dims, default=all_dims)

    # 展示方式：网格卡片 or 表格
    view_mode = st.sidebar.radio("展示形式", ["网格卡片（推荐）", "表格"], index=0)

    # 过滤：固定 period
    dff = df[df["period"] == period_selected].copy()

    # 过滤：维度集合
    if dims_selected:
        dff = dff[dff["dimension"].isin(dims_selected)]

    # 关键词筛选（在这个 period 内搜）
    dff = apply_keyword_filter(dff, keyword)

    # 维度排序
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
        # 网格卡片：一行 2-3 个卡片更像看板
        # 这里用 2 列布局，阅读更舒服；你想 3 列也可以改
        cols_per_row = st.sidebar.selectbox("网格：每行卡片列数", [2, 3], index=0)

        rows = dff.to_dict(orient="records")
        for i in range(0, len(rows), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j >= len(rows):
                    break
                with cols[j]:
                    render_card(pd.Series(rows[i + j]))
