import re
import time
import io
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="B站运营数据Dashboard", layout="wide")

# =========================
# Constants
# =========================
BASELINE_PROJECT = "__BASELINE__"
DB_PATH = "bili_dashboard.db"
TABLE_NAME = "videos"

# =========================
# DB
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_conn() as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            project TEXT NOT NULL,
            bvid TEXT NOT NULL,
            url TEXT,
            title TEXT,
            pubdate TEXT,
            owner_mid TEXT,
            owner_name TEXT,
            view INTEGER,
            like INTEGER,
            coin INTEGER,
            favorite INTEGER,
            reply INTEGER,
            danmaku INTEGER,
            share INTEGER,
            fans_delta INTEGER,
            baseline_for TEXT,
            data_type TEXT,
            fetched_at TEXT,
            PRIMARY KEY (project, bvid)
        )
        """)
        conn.commit()

def load_all_rows() -> pd.DataFrame:
    init_db()
    with db_conn() as conn:
        return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

def upsert_rows(df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return
    init_db()
    cols = [
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta",
        "baseline_for","data_type","fetched_at"
    ]
    df_new = df_new.copy()
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new[cols]

    records = []
    for _, r in df_new.iterrows():
        records.append(tuple(None if pd.isna(v) else v for v in r.tolist()))

    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {TABLE_NAME} ({colnames}) VALUES ({placeholders})"
    with db_conn() as conn:
        conn.executemany(sql, records)
        conn.commit()

def clear_all_data():
    init_db()
    with db_conn() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME}")
        conn.commit()

# =========================
# Utils
# =========================
NUM_COLS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share", "fans_delta"]
EXTRA_COLS = ["baseline_for", "data_type"]

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
        "基准归属": "baseline_for",
        "数据类型": "data_type",
        "抓取时间": "fetched_at",
    }

    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in zh_alias:
            rename[c] = zh_alias[key]
        else:
            low = key.lower()
            if low in [
                "project","url","bvid","title","owner_name","owner_mid","pubdate",
                "view","like","coin","favorite","reply","danmaku","share","fans_delta",
                "baseline_for","data_type","fetched_at"
            ]:
                rename[c] = low
    df = df.rename(columns=rename)

    if "bvid" not in df.columns and "url" in df.columns:
        df["bvid"] = df["url"].apply(parse_bvid)

    if "bvid" in df.columns:
        df["bvid"] = df["bvid"].apply(lambda x: parse_bvid(x) if isinstance(x, str) else x)
        df["bvid"] = df["bvid"].apply(lambda x: _safe_str(x))

    for col in ["project", "title", "owner_name"]:
        if col not in df.columns:
            df[col] = ""

    for col in EXTRA_COLS:
        if col not in df.columns:
            df[col] = ""

    if "pubdate" not in df.columns:
        df["pubdate"] = pd.NaT
    df["pubdate"] = df["pubdate"].apply(_safe_date)

    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].apply(_safe_int)

    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.Timestamp.now()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce").fillna(pd.Timestamp.now())

    keep = set([
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta",
        "baseline_for","data_type","fetched_at"
    ])
    df = df[[c for c in df.columns if c in keep]].copy()
    if "bvid" in df.columns:
        df = df[df["bvid"].astype(str).str.startswith("BV")]
    return df

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement"] = df["like"] + df["coin"] + df["favorite"] + df["reply"]
    df["engagement_rate"] = np.where(df["view"] > 0, df["engagement"] / df["view"], 0.0)
    df["deep_signal_ratio"] = np.where(
        df["engagement"] > 0, (df["coin"] + df["favorite"]) / df["engagement"], 0.0
    )
    return df

# =========================
# Global performance label (KOL recent baseline)
# =========================
def perf_label(value: float, baseline_values: np.ndarray, ratio_hi: float, ratio_lo: float, min_n: int) -> str:
    baseline_values = baseline_values[~np.isnan(baseline_values)]
    if len(baseline_values) < min_n:
        return "基准不足"
    med = float(np.median(baseline_values))
    ratio = (value / med) if med > 1e-12 else np.inf
    if ratio >= ratio_hi:
        return "超常发挥"
    if ratio <= ratio_lo:
        return "低于预期"
    return "正常发挥"

def build_owner_cache(df_all: pd.DataFrame) -> dict:
    cache = {}
    for up, g in df_all.groupby("owner_name"):
        g2 = g.copy()
        g2 = g2[pd.notna(g2["pubdate"])].sort_values("pubdate", ascending=False)
        cache[up] = g2
    return cache

def recent_baseline(df_owner_sorted_desc: pd.DataFrame, current_bvid: str, col: str, window_n: int,
                    exclude_projects: set | None = None) -> np.ndarray:
    if df_owner_sorted_desc is None or df_owner_sorted_desc.empty:
        return np.array([], dtype=float)
    h = df_owner_sorted_desc[df_owner_sorted_desc["bvid"] != current_bvid]
    if exclude_projects:
        h = h[~h["project"].isin(exclude_projects)]
    if h.empty:
        return np.array([], dtype=float)
    return h.head(window_n)[col].astype(float).to_numpy()

def add_perf_cols(df_show: pd.DataFrame, df_all: pd.DataFrame, window_n: int, min_n: int) -> pd.DataFrame:
    df_show = df_show.copy()
    cache = build_owner_cache(df_all)
    v_labels, er_labels = [], []
    for _, r in df_show.iterrows():
        up = r.get("owner_name", "")
        bvid = r.get("bvid", "")
        owner_hist = cache.get(up, None)
        v_base = recent_baseline(owner_hist, bvid, "view", window_n)
        er_base = recent_baseline(owner_hist, bvid, "engagement_rate", window_n)
        v_labels.append(perf_label(float(r.get("view", 0)), v_base, ratio_hi=1.5, ratio_lo=0.7, min_n=min_n))
        er_labels.append(perf_label(float(r.get("engagement_rate", 0.0)), er_base, ratio_hi=1.3, ratio_lo=0.75, min_n=min_n))
    df_show["播放表现"] = v_labels
    df_show["互动率表现"] = er_labels
    return df_show

# =========================
# Bilibili fetch
# =========================
def fetch_video_stats_by_bvid(bvid: str) -> dict:
    api = "https://api.bilibili.com/x/web-interface/view"
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
        "fans_delta": 0,
        "fetched_at": pd.Timestamp.now(),
    }

def fetch_recent_bvids_by_mid(mid: int, n: int = 20) -> list[str]:
    api = "https://api.bilibili.com/x/space/arc/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(api, params={"mid": mid, "pn": 1, "ps": n, "order": "pubdate"}, headers=headers, timeout=10)
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(j.get("message", "UP公开视频列表接口异常"))
    vlist = (((j.get("data") or {}).get("list") or {}).get("vlist")) or []
    out = []
    for v in vlist:
        bvid = v.get("bvid")
        if bvid:
            out.append(bvid)
    return out

# =========================
# Sidebar - global settings
# =========================
st.sidebar.title("📊 B站运营Dashboard")

st.sidebar.markdown("#### 全局“发挥评价”口径")
baseline_window_n = st.sidebar.slider("KOL近期基准：取最近N条视频", 10, 60, 20, step=5)
baseline_min_n = st.sidebar.slider("最低样本数（不足则显示“基准不足”）", 1, 20, 6, step=1)

st.sidebar.divider()

with st.sidebar.expander("备份/恢复", expanded=False):
    df_export = load_all_rows()
    if not df_export.empty:
        st.download_button(
            "⬇️ 导出备份CSV",
            data=df_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="bili_dashboard_backup.csv",
            mime="text/csv"
        )
    else:
        st.caption("暂无数据可导出")

    uploaded_backup = st.file_uploader("导入备份CSV恢复", type=["csv"])
    if uploaded_backup is not None and st.button("📥 恢复备份到数据库"):
        raw = uploaded_backup.getvalue()
        df_imp = None
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                df_imp = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                df_imp = None
        if df_imp is None:
            st.error("恢复失败：CSV读取失败（建议UTF-8编码）。")
        else:
            df_imp = normalize_df(df_imp)
            if "fetched_at" not in df_imp.columns:
                df_imp["fetched_at"] = pd.Timestamp.now()
            df_imp["pubdate"] = pd.to_datetime(df_imp["pubdate"], errors="coerce")
            df_imp["fetched_at"] = pd.to_datetime(df_imp["fetched_at"], errors="coerce").fillna(pd.Timestamp.now())
            df_imp["pubdate"] = df_imp["pubdate"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_imp["fetched_at"] = df_imp["fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_imp)
            st.success("恢复完成。")
            st.rerun()

with st.sidebar.expander("危险操作：清空全部数据", expanded=False):
    if st.button("🗑️ 清空数据库（不可撤销）"):
        clear_all_data()
        st.success("已清空。")
        st.rerun()

st.sidebar.divider()

# =========================
# Data input
# =========================
mode = st.sidebar.radio("数据来源", ["粘贴链接/BV采集", "上传CSV导入"], index=0)

if mode == "粘贴链接/BV采集":
    project = st.sidebar.text_input("项目名（用于归档）", value="未命名项目")
    links = st.sidebar.text_area("粘贴视频链接/ BV号（每行一个）")
    add_btn = st.sidebar.button("➕ 采集并入库（会永久保存）")

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
                row["data_type"] = "collab"
                rows.append(row)
                ok += 1
                time.sleep(0.35)
            except Exception:
                fail += 1

        if rows:
            df_new = normalize_df(pd.DataFrame(rows))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（已保存）")
        st.rerun()

else:
    default_project = st.sidebar.text_input("缺少 project 列时：默认项目名", value="未命名项目")
    uploaded = st.sidebar.file_uploader("选择CSV文件", type=["csv"])
    import_btn = st.sidebar.button("📥 导入CSV到仪表盘（会永久保存）")

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
                st.sidebar.error("CSV读取失败：建议UTF-8编码。")
            else:
                df_csv = normalize_df(df_csv)
                if "project" not in df_csv.columns:
                    df_csv["project"] = default_project
                df_csv["project"] = df_csv["project"].apply(lambda x: _safe_str(x).strip())
                df_csv.loc[df_csv["project"] == "", "project"] = default_project
                if "data_type" not in df_csv.columns:
                    df_csv["data_type"] = "collab"
                if "fetched_at" not in df_csv.columns:
                    df_csv["fetched_at"] = pd.Timestamp.now()

                df_csv["pubdate"] = pd.to_datetime(df_csv["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_csv["fetched_at"] = pd.to_datetime(df_csv["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")
                upsert_rows(df_csv)

                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（已保存）")
                st.rerun()

# =========================
# Load data
# =========================
df_db = load_all_rows()
df_db = normalize_df(df_db) if not df_db.empty else df_db
st.title("B站日常运营数据 Dashboard")
if df_db.empty:
    st.info("数据库为空：请在左侧采集或导入。")
    st.stop()

df_db = compute_metrics(df_db)

# =========================
# Project filter (hide baseline project)
# =========================
projects = sorted([p for p in df_db["project"].dropna().unique().tolist()
                   if str(p).strip() != "" and p != BASELINE_PROJECT])
sel_projects = st.sidebar.multiselect("选择项目（筛选展示）", projects, default=projects if projects else None)
df_main = df_db[df_db["project"] != BASELINE_PROJECT].copy()
df_f = df_main[df_main["project"].isin(sel_projects)].copy() if sel_projects else df_main.copy()

# =========================
# Add global performance labels
# =========================
df_f = add_perf_cols(df_f, df_db, baseline_window_n, baseline_min_n)

# =========================
# KPI
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# =========================
# Cross project comparison
# =========================
st.subheader("跨项目对比（项目之间谁更强、谁更稳）")
proj_rows = []
for proj, g in df_f.groupby("project"):
    g2 = g.sort_values("view", ascending=False).copy()
    total_view = int(g2["view"].sum())
    total_eng = int(g2["engagement"].sum())
    video_cnt = int(len(g2))
    up_cnt = int(g2["owner_name"].nunique())

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
st.dataframe(
    proj_df.assign(**{
        "互动率中位数": (proj_df["互动率中位数"]*100).map(lambda x: f"{x:.2f}%"),
        "深度信号中位数": (proj_df["深度信号中位数"]*100).map(lambda x: f"{x:.1f}%"),
        "互动率波动(IQR)": (proj_df["互动率波动(IQR)"]*100).map(lambda x: f"{x:.2f}pp"),
        "Top1播放贡献": (proj_df["Top1播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
        "Top3播放贡献": (proj_df["Top3播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
    }),
    use_container_width=True,
    height=260
)

# =========================
# Project table
# =========================
st.divider()
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project","bvid","title","owner_name","pubdate",
    "view","播放表现",
    "engagement_rate","互动率表现",
    "like","coin","favorite","reply",
    "deep_signal_ratio"
]
st.dataframe(df_f[show_cols].sort_values("view", ascending=False), use_container_width=True, height=360)

# =========================
# Top/Bottom
# =========================
st.subheader("Top / Bottom 深挖（含KOL近期基准判断）")
for proj in sel_projects if sel_projects else projects:
    d = df_f[df_f["project"] == proj].sort_values("view", ascending=False)
    if d.empty:
        continue
    top = d.iloc[0]
    bottom = d.iloc[-1]

    st.markdown(f"### 项目：{proj}")
    left, right = st.columns(2)

    def render_card(col, row, tag):
        col.markdown(f"**{tag}：{row['title']}**")
        col.caption(f"UP：{row['owner_name']} ｜ BV：{row['bvid']} ｜ 发布：{row['pubdate']}")
        col.metric("播放", f"{int(row['view']):,}", row["播放表现"])
        col.metric("互动率", f"{row['engagement_rate']*100:.2f}%", row["互动率表现"])
        col.write(f"- 赞/币/藏/评：{int(row['like'])}/{int(row['coin'])}/{int(row['favorite'])}/{int(row['reply'])}")
        col.write(f"- 深度信号占比：{row['deep_signal_ratio']*100:.1f}%")

    render_card(left, top, "🔥 最高播放")
    render_card(right, bottom, "🧊 最低播放")

# =========================
# Box plot
# =========================
st.subheader("互动率分布（项目/UP主快速定位异常）")
fig = px.box(df_f, x="project", y="engagement_rate", points="all", hover_data=["title","owner_name","view"])
st.plotly_chart(fig, use_container_width=True)

# =========================
# Auto insights
# =========================
st.subheader("自动解读（可复制进周报）")
best = df_f.sort_values("view", ascending=False).iloc[0]
worst = df_f.sort_values("view", ascending=True).iloc[0]
insights = [
    f"1）本期最高播放《{best['title']}》{int(best['view']):,}（{best['播放表现']}），互动率 {best['engagement_rate']*100:.2f}%（{best['互动率表现']}）。",
    f"2）最低播放《{worst['title']}》{int(worst['view']):,}（{worst['播放表现']}），互动率 {worst['engagement_rate']*100:.2f}%（{worst['互动率表现']}）。建议检查封面/标题信息密度与投放时段，并加强评论区互动引导。",
]
st.write("\n".join(insights))

# =========================================================
# KOL module (critical improvements)
# =========================================================
st.divider()
st.subheader("KOL合作资料库（独立模块：自动补齐基准，解决“基准不足”）")

with st.expander("KOL模块设置", expanded=False):
    collab_projects = st.multiselect("哪些项目算合作项目", projects, default=sel_projects if sel_projects else projects)
    fetch_n = st.slider("补齐基准：每个KOL抓取最近N条公开视频", 10, 60, baseline_window_n, step=5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)

cA, cB, cC = st.columns([1,1,2])
with cA:
    btn_fill_all = st.button("🧲 一键补齐所有合作KOL的近期基准（强烈推荐）")
with cB:
    btn_build_kol = st.button("📚 生成KOL资料库")
with cC:
    st.caption("提示：先点“补齐基准”，再生成资料库，否则很多UP会显示基准不足。")

if collab_projects:
    collab_df = df_db[df_db["project"].isin(collab_projects)].copy()

    if collab_df.empty:
        st.warning("合作项目下没有数据。")
    else:
        st.caption(f"合作UP主数：{collab_df['owner_name'].nunique()}｜合作视频数：{len(collab_df)}")

    # ---- Fill baseline for all KOLs ----
    if btn_fill_all:
        existed = set(df_db["bvid"].astype(str).tolist())
        rows_new = []
        no_mid = 0

        for up, g in collab_df.groupby("owner_name"):
            mids = g["owner_mid"].dropna().unique().tolist()
            if not mids:
                no_mid += 1
                continue
            mid = int(mids[0])

            try:
                bvids = fetch_recent_bvids_by_mid(mid, n=int(fetch_n))
            except Exception:
                continue

            for bvid in bvids:
                if bvid in existed:
                    continue
                try:
                    row = fetch_video_stats_by_bvid(bvid)
                    row["project"] = BASELINE_PROJECT
                    row["baseline_for"] = up
                    row["data_type"] = "baseline"
                    row["url"] = f"https://www.bilibili.com/video/{bvid}"
                    rows_new.append(row)
                    existed.add(bvid)
                    time.sleep(float(sleep_sec))
                except Exception:
                    continue

        if no_mid > 0:
            st.warning(f"有 {no_mid} 位UP缺少 owner_mid，无法自动抓基准。建议：用“链接/BV采集”方式采合作视频（会带owner_mid），或CSV补 owner_mid。")

        if rows_new:
            df_new = normalize_df(pd.DataFrame(rows_new))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)
            st.success(f"已补齐并保存基准样本：新增 {len(rows_new)} 条")
            st.rerun()
        else:
            st.warning("本次未新增基准样本（可能已存在/限流/接口波动）。")

    # ---- Diagnosis: baseline availability ----
    st.markdown("**KOL基准诊断（现在到底缺不缺基准）**")
    df_all = df_db.copy()
    diag = []
    for up, g in collab_df.groupby("owner_name"):
        # baseline candidates: __BASELINE__ + 非合作项目
        base = df_all[(df_all["owner_name"] == up) & (~df_all["project"].isin(collab_projects))].copy()
        base = pd.concat([base, df_all[(df_all["project"] == BASELINE_PROJECT) & (df_all["baseline_for"] == up)]], ignore_index=True)
        base = base.drop_duplicates(subset=["bvid"], keep="last")
        diag.append({
            "KOL/UP主": up,
            "合作视频数": int(len(g)),
            "可用基准数": int(len(base)),
            "状态": "OK" if len(base) >= baseline_min_n else f"基准不足(<{baseline_min_n})",
            "是否有owner_mid": "有" if g["owner_mid"].notna().any() else "无"
        })
    diag_df = pd.DataFrame(diag).sort_values(["状态","可用基准数"], ascending=[True, False])
    st.dataframe(diag_df, use_container_width=True, height=280)

    # ---- Build KOL library ----
    if btn_build_kol:
        df_all_m = compute_metrics(df_db.copy())
        rows = []

        for up, g_collab in df_all_m[df_all_m["project"].isin(collab_projects)].groupby("owner_name"):
            # baseline pool = 非合作 + __BASELINE__
            base_pool = df_all_m[(df_all_m["owner_name"] == up) & (~df_all_m["project"].isin(collab_projects))].copy()
            base_pool = pd.concat([base_pool, df_all_m[(df_all_m["project"] == BASELINE_PROJECT) & (df_all_m["baseline_for"] == up)]], ignore_index=True)
            base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")

            # recent baseline = latest N in baseline_pool
            base_pool = base_pool[pd.notna(base_pool["pubdate"])].sort_values("pubdate", ascending=False)
            recent = base_pool.head(baseline_window_n)

            if len(recent) < baseline_min_n:
                continue

            base_view = float(recent["view"].median())
            base_er = float(recent["engagement_rate"].median())
            base_deep = float(recent["deep_signal_ratio"].median())

            collab_view = float(g_collab["view"].median())
            collab_er = float(g_collab["engagement_rate"].median())
            collab_deep = float(g_collab["deep_signal_ratio"].median())

            view_lift = (collab_view / base_view - 1.0) if base_view > 0 else np.nan
            er_lift = (collab_er / base_er - 1.0) if base_er > 0 else np.nan
            deep_lift = (collab_deep / base_deep - 1.0) if base_deep > 0 else np.nan

            tags = []
            if not np.isnan(view_lift) and view_lift >= 0.30: tags.append("热度拉升型")
            if not np.isnan(er_lift) and er_lift >= 0.20: tags.append("强互动引爆")
            if not np.isnan(deep_lift) and deep_lift >= 0.10: tags.append("价值沉淀型")
            if not tags: tags.append("常规表现")

            persona = f"{'热度拉升' if (not np.isnan(view_lift) and view_lift>=0.3) else '热度稳定'} + {'互动强' if (not np.isnan(er_lift) and er_lift>=0.2) else '互动常规'} + {'沉淀强' if (not np.isnan(deep_lift) and deep_lift>=0.1) else '沉淀一般'}"
            suggest = "适合场景：大促/口碑/试水｜合作形式：测评/系列/软植入｜抓手：卖点前置+互动任务+收藏理由｜避坑：避免硬广直给"

            rows.append({
                "KOL/UP主": up,
                "合作视频数": int(len(g_collab)),
                "基准样本数": int(len(recent)),
                "标签": "、".join(tags),
                "KOL画像一句话": persona,
                "合作建议组合": suggest,
                "合作播放中位数": int(collab_view),
                "基准播放中位数": int(base_view),
                "播放提升": "-" if np.isnan(view_lift) else f"{view_lift*100:.1f}%",
                "合作互动率中位数": f"{collab_er*100:.2f}%",
                "基准互动率中位数": f"{base_er*100:.2f}%",
                "互动率提升": "-" if np.isnan(er_lift) else f"{er_lift*100:.1f}%",
            })

        if not rows:
            st.warning("没有生成任何KOL结果：请先点“一键补齐基准”。")
        else:
            lib = pd.DataFrame(rows).sort_values(["播放提升","互动率提升"], ascending=False)
            st.dataframe(lib, use_container_width=True, height=420)
            st.download_button(
                "⬇️ 下载KOL商务资料库（CSV）",
                data=lib.to_csv(index=False).encode("utf-8-sig"),
                file_name="kol_business_library.csv",
                mime="text/csv"
            )
