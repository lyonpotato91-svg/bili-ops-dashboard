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
NUM_COLS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share", "fans_delta"]

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
    try:
        if pd.isna(x):
            return pd.NaT
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns + dtypes, and ensure required columns exist."""
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

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

    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in zh_alias:
            rename[c] = zh_alias[key]
        else:
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

    # Dates
    if "pubdate" not in df.columns:
        df["pubdate"] = pd.NaT
    df["pubdate"] = df["pubdate"].apply(_safe_date)

    # Numerics
    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].apply(_safe_int)

    # fetched_at
    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.Timestamp.now()
    else:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df["fetched_at"] = df["fetched_at"].fillna(pd.Timestamp.now())

    keep = set(["project","bvid","url","title","pubdate","owner_mid","owner_name",
                "view","like","coin","favorite","reply","danmaku","share","fans_delta","fetched_at"])
    existing = [c for c in df.columns if c in keep]
    df = df[existing].copy()

    # best-effort filter valid BV
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

def _pct(x: float, digits=2):
    return f"{x*100:.{digits}f}%"

# =========================
# B站抓取（链接/BV采集）
# =========================
def fetch_video_stats_by_bvid(bvid: str) -> dict:
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
    merged = merged.sort_values("fetched_at", ascending=True)
    merged = merged.drop_duplicates(subset=["project", "bvid"], keep="last")

    st.session_state["rows"] = merged.to_dict("records")

# =========================
# Sidebar UI
# =========================
st.sidebar.title("📊 B站运营Dashboard")

mode = st.sidebar.radio("数据来源", ["粘贴链接/BV采集", "上传CSV导入"], index=0)

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

st.title("B站日常运营数据 Dashboard")

if df.empty:
    st.info("左侧选择数据来源：粘贴BV/链接采集 或 上传CSV导入。")
    st.stop()

df = compute_metrics(df)

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

# =========================================================
# ✅ 跨项目对比（排行榜 + 四象限）
# =========================================================
st.subheader("跨项目对比（项目之间谁更强、谁更稳）")

proj_rows = []
for proj, g in df_f.groupby("project"):
    g2 = g.sort_values("view", ascending=False).copy()
    total_view = int(g2["view"].sum())
    total_eng = int(g2["engagement"].sum())
    video_cnt = int(len(g2))
    up_cnt = int(g2["owner_name"].nunique()) if "owner_name" in g2.columns else 0

    er_med = float(g2["engagement_rate"].median())
    deep_med = float(g2["deep_signal_ratio"].median())

    er_q1 = float(g2["engagement_rate"].quantile(0.25))
    er_q3 = float(g2["engagement_rate"].quantile(0.75))
    er_iqr = er_q3 - er_q1

    top1_view = int(g2.iloc[0]["view"]) if video_cnt > 0 else 0
    top3_view = int(g2.head(3)["view"].sum()) if video_cnt > 0 else 0
    top1_share = (top1_view / total_view) if total_view > 0 else 0.0
    top3_share = (top3_view / total_view) if total_view > 0 else 0.0

    proj_rows.append({
        "project": proj,
        "视频数": video_cnt,
        "UP数": up_cnt,
        "总播放": total_view,
        "总互动": total_eng,
        "互动率中位数": er_med,
        "深度信号中位数": deep_med,
        "互动率波动(IQR)": er_iqr,
        "Top1播放贡献": top1_share,
        "Top3播放贡献": top3_share,
    })

proj_df = pd.DataFrame(proj_rows).sort_values("总播放", ascending=False)

show_proj_cols = ["project","视频数","UP数","总播放","总互动","互动率中位数","深度信号中位数","互动率波动(IQR)","Top1播放贡献","Top3播放贡献"]
st.dataframe(
    proj_df[show_proj_cols]
      .assign(**{
          "互动率中位数": (proj_df["互动率中位数"]*100).map(lambda x: f"{x:.2f}%"),
          "深度信号中位数": (proj_df["深度信号中位数"]*100).map(lambda x: f"{x:.1f}%"),
          "互动率波动(IQR)": (proj_df["互动率波动(IQR)"]*100).map(lambda x: f"{x:.2f}pp"),
          "Top1播放贡献": (proj_df["Top1播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
          "Top3播放贡献": (proj_df["Top3播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
      }),
    use_container_width=True,
    height=260
)

st.markdown("**项目四象限（X=互动率中位数，Y=深度信号中位数）**")
if len(proj_df) >= 2:
    x_med = float(proj_df["互动率中位数"].median())
    y_med = float(proj_df["深度信号中位数"].median())

    fig2 = px.scatter(
        proj_df,
        x="互动率中位数",
        y="深度信号中位数",
        size="总播放",
        hover_data=["视频数","UP数","总播放","Top1播放贡献","Top3播放贡献","互动率波动(IQR)"],
        text="project",
    )
    fig2.add_vline(x=x_med, line_dash="dash")
    fig2.add_hline(y=y_med, line_dash="dash")
    fig2.update_traces(textposition="top center")
    fig2.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**项目解读（可直接写进周报）**")
    bullets = []
    for _, r in proj_df.iterrows():
        proj = r["project"]
        er = r["互动率中位数"]
        deep = r["深度信号中位数"]
        top1 = r["Top1播放贡献"]
        iqr = r["互动率波动(IQR)"]

        if er >= x_med and deep >= y_med:
            tag = "又热又沉淀（优先加码）"
        elif er >= x_med and deep < y_med:
            tag = "热闹但偏浅（强化收藏/投币引导）"
        elif er < x_med and deep >= y_med:
            tag = "小众但真爱（适合系列化/精准投放）"
        else:
            tag = "偏弱（重点复盘选题&包装&分发）"

        extra = []
        if top1 >= 0.45:
            extra.append("头部依赖高（Top1贡献偏大，需补中腰部）")
        if len(proj_df) > 0 and iqr >= float(proj_df["互动率波动(IQR)"].median()):
            extra.append("波动较大（内容/分发不稳定）")

        extra_txt = f"；{ '；'.join(extra) }" if extra else ""
        bullets.append(
            f"- 互动率中位数 {er*100:.2f}%，深度信号中位数 {deep*100:.1f}%：{tag}{extra_txt}"
        )
    st.write("\n".join(bullets))
else:
    st.info("项目数不足（<2）时，四象限对比意义不大。再导入/采集至少两个项目的数据即可展示。")

st.divider()

# =========================================================
# ✅ 新增：KOL合作效果标注（合作视频明显强于其日常）
# =========================================================
st.subheader("KOL合作效果标注（为未来合作资料库做储备）")

all_projects = sorted([p for p in df["project"].dropna().unique().tolist() if str(p).strip() != ""])
default_collab = sel_projects if sel_projects else all_projects

with st.expander("设置：如何判定“合作明显更好”（可不改，默认就能用）", expanded=False):
    collab_projects = st.multiselect("哪些项目算“合作项目”", all_projects, default=default_collab)
    baseline_mode = st.radio(
        "日常基准怎么取？",
        ["优先用“非合作项目”的视频作为基准（推荐）", "直接用该UP在你库里的所有视频作为基准"],
        index=0
    )
    min_baseline_n = st.slider("基准最少需要多少条视频（太少不判定）", 3, 30, 6)
    lift_view_pct = st.slider("合作播放提升阈值（相对基准中位数）", 0, 300, 30, step=5)  # %
    lift_er_pct = st.slider("合作互动率提升阈值（相对基准中位数）", 0, 300, 20, step=5)   # %
    z_threshold = st.slider("Z分数阈值（用于“明显更好”标注）", 0.0, 3.0, 1.0, step=0.1)
    require_both = st.checkbox("更严格：播放&互动率都要明显更好才标注", value=False)

# 如果用户一个都没选，就不计算
if not collab_projects:
    st.info("请先在设置里选择至少一个“合作项目”。")
else:
    df_all = df.copy()
    df_all = df_all[df_all["owner_name"].astype(str).str.strip() != ""].copy()

    collab_df = df_all[df_all["project"].isin(collab_projects)].copy()

    if collab_df.empty:
        st.info("当前选择的合作项目下没有视频数据。")
    else:
        # baseline: per UP
        rows = []
        for up, g_collab in collab_df.groupby("owner_name"):
            # baseline selection
            if baseline_mode.startswith("优先用"):
                g_base = df_all[(df_all["owner_name"] == up) & (~df_all["project"].isin(collab_projects))].copy()
                # 如果非合作基准太少，兜底用全库
                if len(g_base) < min_baseline_n:
                    g_base = df_all[df_all["owner_name"] == up].copy()
            else:
                g_base = df_all[df_all["owner_name"] == up].copy()

            # 基准数量检查
            if len(g_base) < min_baseline_n:
                continue

            # metrics
            base_view_med = float(g_base["view"].median())
            base_er_med = float(g_base["engagement_rate"].median())
            base_view_mean = float(g_base["view"].mean())
            base_view_std = float(g_base["view"].std(ddof=0)) if float(g_base["view"].std(ddof=0)) > 1e-9 else 0.0
            base_er_mean = float(g_base["engagement_rate"].mean())
            base_er_std = float(g_base["engagement_rate"].std(ddof=0)) if float(g_base["engagement_rate"].std(ddof=0)) > 1e-12 else 0.0

            collab_view_med = float(g_collab["view"].median())
            collab_er_med = float(g_collab["engagement_rate"].median())

            # lifts (relative to baseline median)
            view_lift = (collab_view_med / base_view_med - 1.0) if base_view_med > 0 else np.nan
            er_lift = (collab_er_med / base_er_med - 1.0) if base_er_med > 0 else np.nan

            # z-score using baseline mean/std
            z_view = (collab_view_med - base_view_mean) / base_view_std if base_view_std > 0 else 0.0
            z_er = (collab_er_med - base_er_mean) / base_er_std if base_er_std > 0 else 0.0

            # decide "明显更好"
            cond_view = (not np.isnan(view_lift)) and (view_lift >= lift_view_pct / 100.0) and (z_view >= z_threshold)
            cond_er = (not np.isnan(er_lift)) and (er_lift >= lift_er_pct / 100.0) and (z_er >= z_threshold)

            if require_both:
                is_good = cond_view and cond_er
            else:
                is_good = cond_view or cond_er

            # top collab videos titles
            g_top = g_collab.sort_values("view", ascending=False).head(3)
            top_titles = "｜".join([str(t)[:30] for t in g_top["title"].tolist()])

            rows.append({
                "KOL/UP主": up,
                "合作视频数": int(len(g_collab)),
                "基准视频数": int(len(g_base)),
                "合作播放中位数": collab_view_med,
                "日常播放中位数": base_view_med,
                "合作播放提升": view_lift,
                "合作互动率中位数": collab_er_med,
                "日常互动率中位数": base_er_med,
                "合作互动率提升": er_lift,
                "Z(播放)": z_view,
                "Z(互动率)": z_er,
                "标注": "⭐ 合作明显更好" if is_good else "",
                "合作Top3视频标题": top_titles
            })

        if not rows:
            st.info("暂无满足“基准视频数量”条件的KOL可计算（可降低基准最少条数或补充更多日常视频数据）。")
        else:
            kol_df = pd.DataFrame(rows)

            # 排序：先标注、再按互动率提升、再按播放提升
            kol_df["_flag"] = kol_df["标注"].apply(lambda x: 1 if str(x).strip() else 0)
            kol_df = kol_df.sort_values(["_flag", "合作互动率提升", "合作播放提升"], ascending=[False, False, False]).drop(columns=["_flag"])

            # 展示用格式化
            show = kol_df.copy()
            show["合作播放中位数"] = show["合作播放中位数"].map(lambda x: f"{int(x):,}")
            show["日常播放中位数"] = show["日常播放中位数"].map(lambda x: f"{int(x):,}")
            show["合作播放提升"] = show["合作播放提升"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")
            show["合作互动率中位数"] = show["合作互动率中位数"].map(lambda x: f"{x*100:.2f}%")
            show["日常互动率中位数"] = show["日常互动率中位数"].map(lambda x: f"{x*100:.2f}%")
            show["合作互动率提升"] = show["合作互动率提升"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")
            show["Z(播放)"] = show["Z(播放)"].map(lambda x: f"{x:.2f}")
            show["Z(互动率)"] = show["Z(互动率)"].map(lambda x: f"{x:.2f}")

            st.markdown("**标注逻辑：当合作表现相对该UP日常基准显著提升（提升阈值 + Z分数阈值）时，打上“⭐ 合作明显更好”。**")
            st.dataframe(show, use_container_width=True, height=360)

            # 下载：KOL资料库
            export = kol_df.copy()
            export["合作播放提升"] = export["合作播放提升"].fillna(0.0)
            export["合作互动率提升"] = export["合作互动率提升"].fillna(0.0)
            out_bytes = export.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 下载KOL合作效果资料库（CSV）", data=out_bytes, file_name="kol_collab_performance_library.csv", mime="text/csv")

            # 简短结论
            flagged = kol_df[kol_df["标注"].astype(str).str.contains("⭐")]
            if not flagged.empty:
                tops = flagged.head(5)["KOL/UP主"].tolist()
                st.success(f"已标注 {len(flagged)} 位“合作明显更好”的KOL。优先关注：{ '、'.join(tops) }")
            else:
                st.warning("当前阈值下暂无“合作明显更好”的KOL。你可以在上方设置里适当降低提升阈值或Z阈值。")

st.divider()

# =========================
# 项目内视频表
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

if "fans_delta" in df_f.columns and df_f["fans_delta"].abs().sum() > 0:
    total_fd = int(df_f["fans_delta"].sum())
    insights.append(f"4）项目口径下粉丝净增合计：{total_fd:,}（如该列来自CSV/快照口径，可作为转粉效率复盘依据）。")

st.write("\n".join(insights))
