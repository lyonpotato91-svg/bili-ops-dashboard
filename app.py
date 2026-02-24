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
BASELINE_PROJECT = "__BASELINE__"       # 隐藏项目：不出现在项目归档/筛选里
DB_PATH = "bili_dashboard.db"           # SQLite文件（持久化）
TABLE_NAME = "videos"

# =========================
# DB: init / read / upsert
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
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    return df

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
    df["coin_rate"] = np.where(df["view"] > 0, df["coin"] / df["view"], 0.0)
    df["fav_rate"] = np.where(df["view"] > 0, df["favorite"] / df["view"], 0.0)
    df["reply_rate"] = np.where(df["view"] > 0, df["reply"] / df["view"], 0.0)
    df["deep_signal_ratio"] = np.where(
        df["engagement"] > 0, (df["coin"] + df["favorite"]) / df["engagement"], 0.0
    )
    return df

# =========================
# ✅ 全局评价逻辑（核心升级点）
# - 以“KOL本人近期视频”为基准（默认近20条）
# - 样本不足 -> “基准不足”
# - 播放/互动率可分别配置倍率阈值
# =========================
def performance_label(value: float,
                      baseline_values: np.ndarray,
                      ratio_hi: float,
                      ratio_lo: float,
                      z_hi: float,
                      z_lo: float,
                      min_n: int) -> str:
    baseline_values = baseline_values[~np.isnan(baseline_values)]
    if len(baseline_values) < min_n:
        return "基准不足"
    med = float(np.median(baseline_values))
    mean = float(np.mean(baseline_values))
    std = float(np.std(baseline_values, ddof=0))
    ratio = (value / med) if med > 1e-12 else np.inf
    z = (value - mean) / std if std > 1e-12 else 0.0

    if (ratio >= ratio_hi) or (z >= z_hi):
        return "超常发挥"
    if (ratio <= ratio_lo) or (z <= z_lo):
        return "低于预期"
    return "正常发挥"

def build_owner_history_cache(df_all: pd.DataFrame) -> dict:
    """
    cache[owner_name] = df_owner_sorted_by_pubdate_asc
    """
    cache = {}
    for up, g in df_all.groupby("owner_name"):
        g2 = g.copy()
        g2 = g2[pd.notna(g2["pubdate"])]
        g2 = g2.sort_values("pubdate", ascending=True)
        cache[up] = g2
    return cache

def recent_baseline_values(df_owner_sorted: pd.DataFrame,
                           current_pubdate: pd.Timestamp,
                           col: str,
                           window_n: int) -> np.ndarray:
    """
    取该UP在 current_pubdate 之前的最近 window_n 条数据作为基准。
    """
    if df_owner_sorted is None or df_owner_sorted.empty or pd.isna(current_pubdate):
        return np.array([], dtype=float)

    # 只取更早发布的内容作为“近期基准”
    hist = df_owner_sorted[df_owner_sorted["pubdate"] < current_pubdate]
    if hist.empty:
        return np.array([], dtype=float)

    tail = hist.tail(window_n)
    return tail[col].astype(float).to_numpy()

def add_performance_columns(df_show: pd.DataFrame,
                            df_all_for_baseline: pd.DataFrame,
                            window_n: int,
                            min_n: int) -> pd.DataFrame:
    """
    给展示用df新增：
    - view_perf
    - er_perf
    """
    df_show = df_show.copy()
    df_all_for_baseline = df_all_for_baseline.copy()

    # 用“全库”（含__BASELINE__）作为个人历史基准更合理
    cache = build_owner_history_cache(df_all_for_baseline)

    view_labels = []
    er_labels = []

    for _, r in df_show.iterrows():
        up = r.get("owner_name", "")
        pub = r.get("pubdate", pd.NaT)

        df_owner = cache.get(up, None)

        # 播放：阈值偏严格一点（爆点更容易被识别出来）
        view_base = recent_baseline_values(df_owner, pub, "view", window_n)
        view_labels.append(
            performance_label(
                float(r.get("view", 0)),
                view_base,
                ratio_hi=1.5, ratio_lo=0.7,
                z_hi=1.0, z_lo=-1.0,
                min_n=min_n
            )
        )

        # 互动率：倍率阈值略温和
        er_base = recent_baseline_values(df_owner, pub, "engagement_rate", window_n)
        er_labels.append(
            performance_label(
                float(r.get("engagement_rate", 0.0)),
                er_base,
                ratio_hi=1.3, ratio_lo=0.75,
                z_hi=1.0, z_lo=-1.0,
                min_n=min_n
            )
        )

    df_show["播放表现"] = view_labels
    df_show["互动率表现"] = er_labels
    return df_show

# =========================
# B站抓取
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

def fetch_recent_bvids_by_mid(mid: int, n: int = 5) -> list[str]:
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
# Sidebar: global baseline settings
# =========================
st.sidebar.title("📊 B站运营Dashboard")

st.sidebar.markdown("#### 全局“发挥评价”口径（已升级）")
baseline_window_n = st.sidebar.slider("KOL近期基准：取最近N条视频", 5, 60, 20, step=5)
baseline_min_n = st.sidebar.slider("最低样本数（不足则显示“基准不足”）", 1, 20, 8, step=1)
st.sidebar.caption("说明：所有“正常/超常/低于预期”都基于该UP主近期视频对比，而不是项目内对比。")

st.sidebar.divider()

# =========================
# Sidebar: persistence controls
# =========================
st.sidebar.markdown("#### 数据保存（刷新/换设备不丢）")

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
            st.success("恢复完成（已写入数据库）。")
            st.rerun()

with st.sidebar.expander("危险操作：清空全部数据", expanded=False):
    if st.button("🗑️ 清空数据库（不可撤销）"):
        clear_all_data()
        st.success("已清空。")
        st.rerun()

st.sidebar.divider()

# =========================
# Data Source UI
# =========================
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
st.sidebar.download_button(
    "下载CSV模板",
    data=template_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="bili_dashboard_template.csv",
    mime="text/csv"
)

st.sidebar.divider()

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
                time.sleep(0.4)
            except Exception:
                fail += 1

        if rows:
            df_new = normalize_df(pd.DataFrame(rows))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（数据已保存）")
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
                st.sidebar.error("CSV读取失败：请确认文件编码（建议UTF-8）或格式正确。")
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
                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（数据已保存）")
                st.rerun()

# =========================
# Load data from DB (always)
# =========================
df_db = load_all_rows()
df_db = normalize_df(df_db) if not df_db.empty else df_db

st.title("B站日常运营数据 Dashboard")

if df_db.empty:
    st.info("当前数据库为空：请在左侧采集链接/BV 或 上传CSV导入。数据会永久保存。")
    st.stop()

df_db = compute_metrics(df_db)

# =========================
# Project filter (hide baseline project)
# =========================
projects = sorted([p for p in df_db["project"].dropna().unique().tolist() if str(p).strip() != "" and p != BASELINE_PROJECT])
sel_projects = st.sidebar.multiselect("选择项目（筛选展示）", projects, default=projects if projects else None)

df_main = df_db[df_db["project"] != BASELINE_PROJECT].copy()
df_f = df_main[df_main["project"].isin(sel_projects)].copy() if sel_projects else df_main.copy()

# =========================
# ✅ 给主展示df加入“播放表现/互动率表现”（全局口径）
# 用全库df_db（含__BASELINE__）做个人近期基准
# =========================
df_f = add_performance_columns(
    df_show=df_f,
    df_all_for_baseline=df_db,
    window_n=baseline_window_n,
    min_n=baseline_min_n
)

# =========================
# KPI cards
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# =========================
# Cross project comparison + quadrant
# =========================
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

# =========================
# Project table (now includes performance labels)
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
st.dataframe(
    df_f[show_cols].sort_values("view", ascending=False),
    use_container_width=True,
    height=360
)

# =========================
# Top/Bottom Deep dive (use new labels)
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
insights = []
insights.append(
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放，{best['播放表现']}），互动率 {best['engagement_rate']*100:.2f}%（{best['互动率表现']}）。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放，{worst['播放表现']}），互动率 {worst['engagement_rate']*100:.2f}%（{worst['互动率表现']}）。建议优先检查封面/标题信息密度与投放时段，并在评论区做更强的互动引导。"
)
if df_f["deep_signal_ratio"].mean() < 0.35:
    insights.append("3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”，建议强化：价值点前置、结尾引导收藏/投币、增加系列化承诺。")
else:
    insights.append("3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性，可考虑围绕该方向做系列化与固定栏目节奏。")
st.write("\n".join(insights))

# =========================================================
# KOL module (independent) + Diagnosis
# （KOL模块也会受全局“近期基准口径”影响，因为baseline数据会进入df_db）
# =========================================================
st.divider()
st.subheader("KOL合作资料库（独立模块：含诊断）")

all_projects = projects
default_collab = sel_projects if sel_projects else all_projects

with st.expander("KOL模块设置（默认即可）", expanded=False):
    collab_projects = st.multiselect("哪些项目算“合作项目”", all_projects, default=default_collab)
    extra_baseline_n = st.slider("自动补齐：每个KOL额外抓几条日常视频", 0, 30, 10)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)

cA, cB = st.columns(2)
with cA:
    fetch_baseline_btn = st.button("🧲 自动抓KOL日常样本（保存到库）")
with cB:
    build_kol_btn = st.button("📚 生成KOL资料库（基于全局近期口径）")

if collab_projects:
    collab_df = df_db[df_db["project"].isin(collab_projects)].copy()

    if collab_df.empty:
        st.warning("合作项目下没有数据：请确认项目名与数据里的 project 完全一致。")
    else:
        st.caption(f"合作UP主数：{collab_df['owner_name'].nunique()}｜合作视频数：{len(collab_df)}")

    if fetch_baseline_btn:
        if collab_df.empty or extra_baseline_n <= 0:
            st.warning("合作项目下没有数据，或补齐数量为0。")
        else:
            existed = set(df_db["bvid"].astype(str).tolist())
            rows_new = []
            failed_no_mid = 0

            for up, g in collab_df.groupby("owner_name"):
                mids = g["owner_mid"].dropna().unique().tolist()
                if not mids:
                    failed_no_mid += 1
                    continue
                mid = int(mids[0])

                try:
                    recent_bvids = fetch_recent_bvids_by_mid(mid, n=int(extra_baseline_n))
                except Exception:
                    continue

                for bvid in recent_bvids:
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

            if failed_no_mid > 0:
                st.warning(
                    f"有 {failed_no_mid} 位UP缺少 owner_mid，无法自动抓日常样本。"
                    "建议：用“链接/BV采集”方式采合作视频（会带owner_mid），或CSV补 owner_mid。"
                )

            if rows_new:
                df_new = normalize_df(pd.DataFrame(rows_new))
                df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                upsert_rows(df_new)
                st.success(f"已保存KOL日常样本：新增 {len(rows_new)} 条")
                st.rerun()
            else:
                st.warning("未抓到可新增的日常样本（可能限流/接口波动/样本已存在）。")

    # 诊断表：哪个KOL的近期基准不足
    st.markdown("**KOL基准诊断（谁的近期样本不足）**")
    df_all = df_db.copy()
    diag = []
    for up, g in collab_df.groupby("owner_name"):
        owner_all = df_all[df_all["owner_name"] == up].copy()
        owner_all = owner_all[pd.notna(owner_all["pubdate"])].sort_values("pubdate", ascending=True)
        # “近期基准”样本量：取最后 baseline_window_n 条（不区分合作/非合作）
        base_n = int(min(len(owner_all), baseline_window_n))
        diag.append({
            "KOL/UP主": up,
            "库内视频数": int(len(owner_all)),
            "可用于近期基准的样本数": base_n,
            "状态": "基准不足" if base_n < baseline_min_n else "OK"
        })
    diag_df = pd.DataFrame(diag).sort_values(["状态","库内视频数"], ascending=[True, False])
    st.dataframe(diag_df, use_container_width=True, height=260)

    # 生成KOL库：以合作视频相对“该UP近期基准”的提升来给标签
    if build_kol_btn:
        df_all_m = compute_metrics(df_db.copy())
        rows = []
        for up, g_collab in df_all_m[df_all_m["project"].isin(collab_projects)].groupby("owner_name"):
            owner_all = df_all_m[df_all_m["owner_name"] == up].copy()
            owner_all = owner_all[pd.notna(owner_all["pubdate"])].sort_values("pubdate", ascending=True)

            # 用该UP“近期基准”的中位数做对比（简单稳定）
            recent = owner_all.tail(baseline_window_n)
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

            # 商务标签
            tags = []
            if not np.isnan(view_lift) and view_lift >= 0.30: tags.append("热度拉升型")
            if not np.isnan(er_lift) and er_lift >= 0.20: tags.append("强互动引爆")
            if not np.isnan(deep_lift) and deep_lift >= 0.10: tags.append("价值沉淀型")
            if not tags: tags.append("常规表现")

            persona = f"{'热度拉升' if (not np.isnan(view_lift) and view_lift>=0.3) else '热度稳定'} + {'互动强' if (not np.isnan(er_lift) and er_lift>=0.2) else '互动常规'} + {'沉淀强' if (not np.isnan(deep_lift) and deep_lift>=0.1) else '沉淀一般'}"

            suggestion_bundle = "适合场景：大促/口碑/试水｜合作形式：测评/系列/软植入｜内容抓手：前3秒卖点+互动任务+收藏理由｜避坑：避免硬广直给"

            rows.append({
                "KOL/UP主": up,
                "合作视频数": int(len(g_collab)),
                "近期基准样本数": int(len(recent)),
                "标签": "、".join(tags),
                "KOL画像一句话": persona,
                "合作建议组合": suggestion_bundle,
                "合作播放中位数": int(collab_view),
                "基准播放中位数": int(base_view),
                "播放提升": "-" if np.isnan(view_lift) else f"{view_lift*100:.1f}%",
                "合作互动率中位数": f"{collab_er*100:.2f}%",
                "基准互动率中位数": f"{base_er*100:.2f}%",
                "互动率提升": "-" if np.isnan(er_lift) else f"{er_lift*100:.1f}%",
            })

        if not rows:
            st.warning("没有生成任何KOL结果：多数UP可能“近期基准样本不足”。先补齐样本再生成。")
        else:
            lib = pd.DataFrame(rows).sort_values(["播放提升","互动率提升"], ascending=False)
            st.dataframe(lib, use_container_width=True, height=420)
            st.download_button(
                "⬇️ 下载KOL商务资料库（CSV）",
                data=lib.to_csv(index=False).encode("utf-8-sig"),
                file_name="kol_business_library.csv",
                mime="text/csv"
            )
