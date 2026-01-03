import re
import pandas as pd
import streamlit as st

# ==============================
# 基础配置
# ==============================
st.set_page_config(page_title="Uber Review Insights Dashboard", layout="wide")

# 你导出的 Excel 路径（按需修改）
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
# 读取数据（用缓存加速，避免每次交互都重新读文件）
# ==============================
@st.cache_data(show_spinner=False)
def load_data(path: str, sheet: str):
    df = pd.read_excel(path, sheet_name=sheet)
    # 标准化列名（防止意外空格）
    df.columns = [str(c).strip() for c in df.columns]

    # 必要列检查
    required = {"dimension", "period", "ai_summary", "ai_recommendations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 你当前列为: {df.columns.tolist()}")

    # 生成排序用时间 key
    df["period_sort"] = df["period"].apply(period_to_sortkey)
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

# 如果 period_sort 有 NaT，提示一下（不影响使用，但排序可能有问题）
if df["period_sort"].isna().any():
    st.warning("有部分 period 无法解析为时间（period_sort=NaT），排序可能不准确。请检查 period 格式（YYYYMM 或 YYYYQn）。")

# ==============================
# 侧边栏筛选器
# ==============================
st.sidebar.header("筛选条件")

# 维度选择
all_dims = sorted(df["dimension"].dropna().unique().tolist())
dim_selected = st.sidebar.selectbox("选择维度 (dimension)", options=["(All)"] + all_dims, index=0)

# 时间范围选择（用可排序时间 key）
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

# 关键词搜索（在 summary/reco/证据里搜）
keyword = st.sidebar.text_input("关键词搜索（summary/reco/evidence）", value="").strip()

# 展示模式
view_mode = st.sidebar.radio("展示模式", ["卡片 (按时间排序)", "表格"], index=0)

# 每页显示多少条（卡片模式）
page_size = st.sidebar.selectbox("卡片模式：每页显示条数", [10, 20, 50, 100], index=1)

# ==============================
# 应用筛选
# ==============================
dff = df.copy()

# 按维度筛选
if dim_selected != "(All)":
    dff = dff[dff["dimension"] == dim_selected]

# 按时间范围筛选
if time_range is not None and not df_valid_time.empty:
    start_dt, end_dt = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
    dff = dff.dropna(subset=["period_sort"])
    dff = dff[(dff["period_sort"] >= start_dt) & (dff["period_sort"] <= end_dt)]

# 关键词筛选
if keyword:
    # 拼接一个检索字段（避免多列 contains 写很多次）
    cols_for_search = ["ai_summary", "ai_recommendations"]
    if "negative_reviews_concat" in dff.columns:
        cols_for_search.append("negative_reviews_concat")
    search_text = dff[cols_for_search].fillna("").astype(str).agg("\n".join, axis=1)
    dff = dff[search_text.str.contains(keyword, case=False, na=False)]

# 排序：先维度后时间
dff = dff.sort_values(["dimension", "period_sort", "period"], ascending=[True, True, True])

# ==============================
# 顶部概览
# ==============================
c1, c2, c3 = st.columns(3)
c1.metric("筛选后条目数", len(dff))
c2.metric("维度数", dff["dimension"].nunique())
c3.metric("时间段数", dff["period"].nunique())

st.divider()

# ==============================
# 主区展示
# ==============================
if dff.empty:
    st.info("筛选结果为空。请调整维度/时间范围/关键词。")
    st.stop()

if view_mode.startswith("表格"):
    # 表格模式：直接看全量（适合复制/导出）
    show_cols = ["dimension", "period", "ai_summary", "ai_recommendations"]
    if "negative_reviews_concat" in dff.columns:
        show_cols.append("negative_reviews_concat")
    st.dataframe(dff[show_cols], use_container_width=True, height=700)

else:
    # 卡片模式：分页展示（更像看板）
    # 分页
    total = len(dff)
    page_count = (total + page_size - 1) // page_size
    page = st.number_input("页码", min_value=1, max_value=page_count, value=1, step=1)

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    st.caption(f"显示第 {start+1} - {end} 条 / 共 {total} 条")

    dff_page = dff.iloc[start:end].copy()

    # 逐条渲染卡片
    for _, row in dff_page.iterrows():
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
                st.write(summary if str(summary).strip() else "(empty)")
            with colB:
                st.markdown("**AI Recommendations**")
                st.write(reco if str(reco).strip() else "(empty)")

            if "negative_reviews_concat" in dff_page.columns:
                with st.expander("Evidence (Top negative reviews, concatenated)"):
                    st.text(evidence if str(evidence).strip() else "(empty)")

    st.divider()
    st.caption("提示：左侧可切换维度、时间范围、关键词。卡片模式适合阅读；表格模式适合复制/导出。")
s