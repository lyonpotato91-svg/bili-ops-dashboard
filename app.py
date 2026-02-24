import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="B站运营数据Dashboard", layout="wide")

# ========= 基础：从链接解析 BV 号 =========
def parse_bvid(url_or_bv: str) -> str | None:
    s = (url_or_bv or "").strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", s)
    return m.group(1) if m else None


# ========= 数据抓取（示例：用公开接口思路；你可替换为 CSV/官方平台/内部口径） =========
def fetch_video_stats_by_bvid(bvid: str) -> dict:
    """
    返回字段：title, pubdate, owner_mid, owner_name, view, like, coin, favorite, reply, danmaku, share
    注意：接口可变，这里做演示用；失败会抛错，前端提示改用CSV导入。
    """
    api = "https://api.bilibili.com/x/web-interface/view"

    # ✅ 加 User-Agent，提升稳定性
    headers = {"User-Agent": "Mozilla/5.0"}

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
        "fetched_at": pd.Timestamp.now(),
    }


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
    if baseline_std <= 1e-9:
        return "正常发挥"
    z = (value - baseline_mean) / baseline_std
    if z >= 1.0:
        return "超常发挥"
    if z <= -1.0:
        return "低于预期"
    return "正常发挥"


# ========= 状态存储（简化：用 session；你上线可换 SQLite） =========
if "rows" not in st.session_state:
    st.session_state["rows"] = []


# ========= UI =========
st.sidebar.title("📊 B站运营Dashboard")
project = st.sidebar.text_input("项目名（用于归档）", value="未命名项目")
links = st.sidebar.text_area("粘贴视频链接/ BV号（每行一个）")
add_btn = st.sidebar.button("➕ 采集并入库")

if add_btn:
    items = [x for x in links.splitlines() if x.strip()]
    ok, fail = 0, 0

    for it in items:
        bvid = parse_bvid(it)
        if not bvid:
            fail += 1
            continue

        try:
            row = fetch_video_stats_by_bvid(bvid)
            row["project"] = project
            st.session_state["rows"].append(row)
            ok += 1

            # 适当放慢，减少被限流概率（可自行调大到 0.5 / 1.0）
            time.sleep(0.4)

        except Exception:
            fail += 1

    st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（失败可改用CSV导入兜底）")


df = pd.DataFrame(st.session_state["rows"])
if df.empty:
    st.info("在左侧粘贴视频链接或 BV 号，然后点击“采集并入库”。")
    st.stop()

df = compute_metrics(df)

# 筛选
projects = sorted(df["project"].unique())
sel_projects = st.sidebar.multiselect("选择项目", projects, default=projects)
df_f = df[df["project"].isin(sel_projects)].copy()

st.title("B站日常运营数据 Dashboard")

# KPI 卡片
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# 项目内排行
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project",
    "bvid",
    "title",
    "owner_name",
    "pubdate",
    "view",
    "like",
    "coin",
    "favorite",
    "reply",
    "engagement_rate",
    "deep_signal_ratio",
]
st.dataframe(
    df_f[show_cols].sort_values("view", ascending=False),
    use_container_width=True,
    height=320,
)

# Top/Bottom + 基准对比（同Up的历史基准：用当前库里该Up的所有视频当基准；你以后可扩展为“近30条”）
st.subheader("Top / Bottom 深挖（含Up主基准对比）")
for proj in sel_projects:
    d = df_f[df_f["project"] == proj].sort_values("view", ascending=False)
    if d.empty:
        continue

    top = d.iloc[0]
    bottom = d.iloc[-1]

    st.markdown(f"### 项目：{proj}")
    left, right = st.columns(2)

    def render_card(col, row, tag):
        up = row["owner_name"]
        base = df[df["owner_name"] == up]  # 用当前库里的该UP所有视频做基准

        mean_v, std_v = base["view"].mean(), base["view"].std(ddof=0)
        mean_er, std_er = base["engagement_rate"].mean(), base["engagement_rate"].std(ddof=0)

        col.markdown(f"**{tag}：{row['title']}**")
        col.caption(f"UP：{up} ｜ BV：{row['bvid']} ｜ 发布：{row['pubdate']}")
        col.metric("播放", f"{int(row['view']):,}", label_vs_baseline(row["view"], mean_v, std_v))
        col.metric(
            "互动率",
            f"{row['engagement_rate']*100:.2f}%",
            label_vs_baseline(row["engagement_rate"], mean_er, std_er),
        )
        col.write(
            f"- 赞/币/藏/评：{int(row['like'])}/{int(row['coin'])}/{int(row['favorite'])}/{int(row['reply'])}"
        )
        col.write(f"- 深度信号占比：{row['deep_signal_ratio']*100:.1f}%")

    render_card(left, top, "🔥 最高播放")
    render_card(right, bottom, "🧊 最低播放")

# 互动率分布（快速定位异常）
st.subheader("互动率分布（项目/UP主快速定位异常）")
fig = px.box(df_f, x="project", y="engagement_rate", points="all", hover_data=["title", "owner_name", "view"])
st.plotly_chart(fig, use_container_width=True)

# 自动解读（周报可用）
st.subheader("自动解读（可复制进周报）")
best = df_f.sort_values("view", ascending=False).iloc[0]
worst = df_f.sort_values("view", ascending=True).iloc[0]

insights = []
insights.append(
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放），互动率 {best['engagement_rate']*100:.2f}%，深度信号占比 {best['deep_signal_ratio']*100:.1f}%。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放），互动率 {worst['engagement_rate']*100:.2f}%。建议检查封面/标题信息密度与投放时段，并在评论区做更强的互动引导。"
)

if df_f["deep_signal_ratio"].mean() < 0.35:
    insights.append(
        "3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”，建议强化：价值点前置、结尾引导收藏/投币、增加系列化承诺。"
    )
else:
    insights.append(
        "3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性，可考虑围绕该方向做系列化与固定栏目节奏。"
    )

st.write("\n".join(insights))
