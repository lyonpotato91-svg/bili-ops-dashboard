import re
import time
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="B站运营数据Dashboard", layout="wide")

# =========================
# Utils
# =========================
REQUIRED_COLS_MIN = ["bvid", "title", "owner_name", "pubdate", "view", "like", "coin", "favorite", "reply", "project"]

NUM_COLS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share", "fans_delta"]
OPTIONAL_COLS = ["danmaku", "share", "fans_delta", "owner_mid", "fetched_at", "url"]

def parse_bvid(url_or_bv: str) -> str | None:
    s = (url_or_bv or "").strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", s)
    return m.group(1) if m else None

def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return int(float(x))
    except Exception:
        return default

def _safe_str(x, default=""):
    try:
        if pd.isna(x):
            return default
        return str(x)
    except Exception:
        return default

def _safe_date(x):
    # Accept: YYYY-MM-DD, YYYY/MM/DD, timestamp, etc.
    try:
        if pd.isna(x):
            return pd.NaT
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns + dtypes, and ensure required columns exist."""
    df = df.copy()

    # normalize column names: strip, lower
    df.columns = [str(c).strip() for c in df.columns]
    col_map_lower = {c.lower(): c for c in df.columns}

    # Make sure required columns exist (case-insensitive)
    def pick(col):
        return col_map_lower.get(col, None)

    # Try to map common Chinese headers to our standard schema
    zh_alias = {
        "项目": "project",
        "项目名": "project",
        "视频链接": "url",
        "链接": "url",
        "标题": "title",
        "UP主": "owner_name",
        "UP主名称": "owner_name",
        "发布时间": "pubdate",
        "播放": "view",
        "播放量": "view",
        "点赞": "like",
        "投币": "coin",
        "收藏": "favorite",
        "评论": "reply",
        "弹幕": "danmaku",
        "分享": "share",
        "粉丝增长": "fans_delta",
        "粉丝增量": "fans_delta",
        "BV": "bvid",
        "BV号": "bvid",
        "bvid": "bvid",
    }

    # Build rename dict
    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in zh_alias:
            rename[c] = zh_alias[key]
        else:
            # also handle lowercase match
            low = key.lower()
            if low in ["project","url","bvid","title","owner_name","owner_mid","pubdate",
                       "view","like","coin","favorite","reply","danmaku","share","fans_delta","fetched_at"]:
                rename[c] = low

    df = df.rename(columns=rename)

    # If url exists but bvid missing, try parse
    if "bvid" not in df.columns and "url" in df.columns:
        df["bvid"] = df["url"].apply(parse_bvid)

    # If bvid exists but has URLs inside, parse them
    if "bvid" in df.columns:
        df["bvid"] = df["bvid"].apply(lambda x: parse_bvid(x) if isinstance(x, str) else x)
        df["bvid"] = df["bvid"].apply(lambda x: _safe_str(x))

    # Ensure required text cols exist
    for col in ["project", "title", "owner_name"]:
        if col not in df.columns:
            df[col] = ""

    # Parse dates
    if "pubdate" not in df.columns:
        df["pubdate"] = pd.NaT
    df["pubdate"] = df["pubdate"].apply(_safe_date)

    # Ensure numeric cols exist
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].apply(_safe_int)

    # Ensure fetched_at exists
    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.Timestamp.now()
    else:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df["fetched_at"] = df["fetched_at"].fillna(pd.Timestamp.now())

    # Keep only known cols + required
    keep = set(["project","bvid","url","title","pubdate","owner_mid","owner_name",
                "view","like","coin","favorite","reply","danmaku","share","fans_delta","fetched_at"])
    existing = [c for c in df.columns if c in keep]
    df = df[existing].copy()

    # Drop rows without bvid/title (best effort)
    if "bvid" in df.columns:
        df = df[df["bvid"].astype(str).str.startswith("BV")]

    return df

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement"] = df["like"] + df["coin"] + df["favorite"] + df["reply"]
    df["engagement_rate"] = np.where(df["view"] > 0, df["engagement"] / df["view"], 0.0)
    df["coin_rate"] = np.where(df["view"] > 0, df["coin"] / df["view"], 0.0)
    df["fav_rate"] = np.where(df["view"] > 0, df["favorite"] / df["view"], 0.0)
    df["reply_rate"] = np.where(df["view"] > 0, df["reply"] / df["view"], 0.0)
    df["deep_signal_ratio"] = np.where(
        df["engagement"] > 0, (df["coin"] + df["favorite"]) / df["engagement"], 0.0
    )
    return df

def label_vs_baseline(value: float, baseline_mean: float, baseline_std: float) -> str:
    if baseline_std <= 1e-9 or np.isnan(baseline_std):
        return "正常发挥"
    z = (value - baseline_mean) / baseline_std
    if z >= 1.0:
        return "超常发挥"
    if z <= -1.0:
        return "低于预期"
    return "正常发挥"

# =========================
# B站抓取（链接/BV采集）
# =========================
def fetch_video_stats_by_bvid(bvid: str) -> dict:
    """
    返回字段：title, pubdate, owner_mid, owner_name, view, like, coin, favorite, reply, danmaku, share
    注意：接口可变；失败会抛错，前端提示改用CSV导入兜底。
    """
    api = "https://api.bilibili.com/x/web-interface/view"
    headers = {"User-Agent": "Mozilla/5.0"}  # ✅ 提升稳定性

    r = requests.get(api, params={"bvid": bvid}, headers=headers, timeout=10)
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(data.get("message", "接口返回异常"))

    d = data["data"]
    stat = d.get("stat", {})
    owner = d.get("owner", {})

    return {
        "bvid": bvid,
        "title": d.get("title"),
        "pubdate": pd.to_datetime(d.get("pubdate", 0), unit="s", errors="coerce"),
        "owner_mid": owner.get("mid"),
        "owner_name": owner.get("name"),
        "view": stat.get("view", 0),
        "like": stat.get("like", 0),
        "coin": stat.get("coin", 0),
        "favorite": stat.get("favorite", 0),
        "reply": stat.get("reply", 0),
        "danmaku": stat.get("danmaku", 0),
        "share": stat.get("share", 0),
        "fans_delta": 0,  # 默认0；你后续可用CSV/快照补齐
        "fetched_at": pd.Timestamp.now(),
    }

# =========================
# Session State
# =========================
if "rows" not in st.session_state:
    st.session_state["rows"] = []  # list[dict]

def append_df_to_session(df_new: pd.DataFrame):
    """Append normalized df rows into session_state rows; de-duplicate by (project,bvid)."""
    if df_new is None or df_new.empty:
        return
    existing = pd.DataFrame(st.session_state["rows"])
    if existing.empty:
        st.session_state["rows"] = df_new.to_dict("records")
        return

    existing = normalize_df(existing)
    df_new = normalize_df(df_new)

    merged = pd.concat([existing, df_new], ignore_index=True)
    # 去重：同一项目同一BV，保留最新 fetched_at
    merged = merged.sort_values("fetched_at", ascending=True)
    merged = merged.drop_duplicates(subset=["project", "bvid"], keep="last")

    st.session_state["rows"] = merged.to_dict("records")

# =========================
# Sidebar UI
# =========================
st.sidebar.title("📊 B站运营Dashboard")

mode = st.sidebar.radio("数据来源", ["粘贴链接/BV采集", "上传CSV导入"], index=0)

# ---- CSV 模板下载（可选）
st.sidebar.markdown("#### CSV模板（可选）")
template_df = pd.DataFrame([{
    "project": "枕刀歌",
    "bvid": "BVxxxxxxxxxxx",
    "url": "https://www.bilibili.com/video/BVxxxxxxxxxxx",
    "title": "示例标题",
    "owner_name": "示例UP主",
    "pubdate": "2026-02-01",
    "view": 1566000,
    "like": 52000,
    "coin": 12000,
    "favorite": 18000,
    "reply": 8000,
    "danmaku": 5000,
    "share": 1200,
    "fans_delta": 3200,
}])
csv_bytes = template_df.to_csv(index=False).encode("utf-8-sig")
st.sidebar.download_button("下载CSV模板", data=csv_bytes, file_name="bili_dashboard_template.csv", mime="text/csv")

st.sidebar.divider()

# ---- Mode A: Link/BV collection
if mode == "粘贴链接/BV采集":
    project = st.sidebar.text_input("项目名（用于归档）", value="未命名项目")
    links = st.sidebar.text_area("粘贴视频链接/ BV号（每行一个）")
    add_btn = st.sidebar.button("➕ 采集并入库")

    if add_btn:
        items = [x for x in links.splitlines() if x.strip()]
        ok, fail = 0, 0
        rows = []

        for it in items:
            bvid = parse_bvid(it)
            if not bvid:
                fail += 1
                continue
            try:
                row = fetch_video_stats_by_bvid(bvid)
                row["project"] = project
                row["url"] = it
                rows.append(row)
                ok += 1
                time.sleep(0.4)  # 放慢，降低被限流概率
            except Exception:
                fail += 1

        if rows:
            append_df_to_session(pd.DataFrame(rows))
        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（失败可用CSV导入兜底）")

# ---- Mode B: CSV import
else:
    st.sidebar.markdown("#### 上传CSV并自动归档")
    default_project = st.sidebar.text_input("缺少 project 列时：默认项目名", value="未命名项目")
    uploaded = st.sidebar.file_uploader("选择CSV文件", type=["csv"])

    import_btn = st.sidebar.button("📥 导入CSV到仪表盘")

    if import_btn:
        if not uploaded:
            st.sidebar.error("请先选择一个CSV文件。")
        else:
            raw = uploaded.getvalue()
            # 兼容中文常见编码：utf-8-sig / gbk
            df_csv = None
            for enc in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    df_csv = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    df_csv = None

            if df_csv is None:
                st.sidebar.error("CSV读取失败：请确认文件编码（建议用UTF-8）或格式正确。")
            else:
                df_csv = normalize_df(df_csv)

                # 如果导入后 project 全为空，用默认项目名补齐
                if "project" not in df_csv.columns:
                    df_csv["project"] = default_project
                df_csv["project"] = df_csv["project"].apply(lambda x: _safe_str(x).strip())
                df_csv.loc[df_csv["project"] == "", "project"] = default_project

                append_df_to_session(df_csv)
                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（已按 project 归档/可筛选）")

# =========================
# Main Data
# =========================
df = pd.DataFrame(st.session_state["rows"])
df = normalize_df(df) if not df.empty else df

# Top bar controls
st.title("B站日常运营数据 Dashboard")

if df.empty:
    st.info("左侧选择数据来源：粘贴BV/链接采集 或 上传CSV导入。")
    st.stop()

df = compute_metrics(df)

# Filters
st.sidebar.divider()
projects = sorted([p for p in df["project"].dropna().unique().tolist() if str(p).strip() != ""])
sel_projects = st.sidebar.multiselect("选择项目（筛选展示）", projects, default=projects if projects else None)
df_f = df[df["project"].isin(sel_projects)].copy() if sel_projects else df.copy()

# =========================
# KPI cards
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# =========================
# Table
# =========================
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project", "bvid", "title", "owner_name", "pubdate",
    "view", "like", "coin", "favorite", "reply", "danmaku", "share", "fans_delta",
    "engagement_rate", "deep_signal_ratio"
]
existing_cols = [c for c in show_cols if c in df_f.columns]
st.dataframe(
    df_f[existing_cols].sort_values("view", ascending=False),
    use_container_width=True,
    height=340,
)

# =========================
# Top/Bottom per project + UP baseline compare
# =========================
st.subheader("Top / Bottom 深挖（含Up主基准对比）")
for proj in (sel_projects if sel_projects else projects):
    d = df_f[df_f["project"] == proj].sort_values("view", ascending=False)
    if d.empty:
        continue

    top = d.iloc[0]
    bottom = d.iloc[-1]

    st.markdown(f"### 项目：{proj}")
    left, right = st.columns(2)

    def render_card(col, row, tag):
        up = row.get("owner_name", "")
        base = df[df["owner_name"] == up] if up else df

        mean_v, std_v = base["view"].mean(), base["view"].std(ddof=0)
        mean_er, std_er = base["engagement_rate"].mean(), base["engagement_rate"].std(ddof=0)

        col.markdown(f"**{tag}：{row.get('title','')}**")
        col.caption(f"UP：{up} ｜ BV：{row.get('bvid','')} ｜ 发布：{row.get('pubdate','')}")
        col.metric("播放", f"{int(row.get('view',0)):,}", label_vs_baseline(float(row.get("view",0)), mean_v, std_v))
        col.metric(
            "互动率",
            f"{float(row.get('engagement_rate',0))*100:.2f}%",
            label_vs_baseline(float(row.get("engagement_rate",0)), mean_er, std_er),
        )

        like = int(row.get("like", 0))
        coin = int(row.get("coin", 0))
        fav = int(row.get("favorite", 0))
        rep = int(row.get("reply", 0))
        deep = float(row.get("deep_signal_ratio", 0))*100

        col.write(f"- 赞/币/藏/评：{like:,}/{coin:,}/{fav:,}/{rep:,}")
        col.write(f"- 深度信号占比：{deep:.1f}%")

        # 粉丝增长（如果有）
        if "fans_delta" in df.columns:
            fd = int(row.get("fans_delta", 0))
            col.write(f"- 粉丝净增（如有）：{fd:,}")

    render_card(left, top, "🔥 最高播放")
    render_card(right, bottom, "🧊 最低播放")

# =========================
# Distribution chart
# =========================
st.subheader("互动率分布（项目/UP主快速定位异常）")
fig = px.box(
    df_f,
    x="project",
    y="engagement_rate",
    points="all",
    hover_data=[c for c in ["title", "owner_name", "view"] if c in df_f.columns]
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Auto insights
# =========================
st.subheader("自动解读（可复制进周报）")
best = df_f.sort_values("view", ascending=False).iloc[0]
worst = df_f.sort_values("view", ascending=True).iloc[0]

insights = []
insights.append(
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放），互动率 {best['engagement_rate']*100:.2f}%，深度信号占比 {best['deep_signal_ratio']*100:.1f}%。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放），互动率 {worst['engagement_rate']*100:.2f}%。建议优先检查：封面/标题信息密度、发布时间段、以及评论区置顶引导（提问/投票/福利点）。"
)

deep_mean = df_f["deep_signal_ratio"].mean()
if deep_mean < 0.35:
    insights.append(
        "3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”。建议：价值点前置、结尾强化收藏/投币理由、做系列化承诺（下一期看点）。"
    )
else:
    insights.append(
        "3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性。建议围绕该方向做系列化与固定栏目节奏，提升可预期的复看与关注转化。"
    )

# 粉丝净增（如果可用）
if "fans_delta" in df_f.columns and df_f["fans_delta"].abs().sum() > 0:
    total_fd = int(df_f["fans_delta"].sum())
    insights.append(f"4）项目口径下粉丝净增合计：{total_fd:,}（如该列来自CSV/快照口径，可作为转粉效率复盘依据）。")

st.write("\n".join(insights))
