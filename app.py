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
BASELINE_PROJECT = "__BASELINE__"       # 隐藏项目：不进入项目归档列表
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
    """INSERT OR REPLACE by (project,bvid)."""
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

    # sqlite prefers python scalars
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

    # Keep only known cols
    keep = set([
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta",
        "baseline_for","data_type","fetched_at"
    ])
    df = df[[c for c in df.columns if c in keep]].copy()

    # valid BV only
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
# Sidebar: persistence controls
# =========================
st.sidebar.title("📊 B站运营Dashboard")
st.sidebar.markdown("#### 数据保存")
colx, coly = st.sidebar.columns(2)

with colx:
    if st.button("⬇️ 导出备份CSV"):
        df_export = load_all_rows()
        if df_export.empty:
            st.sidebar.warning("当前没有可导出的数据。")
        else:
            st.download_button(
                "点击下载",
                data=df_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="bili_dashboard_backup.csv",
                mime="text/csv"
            )

with coly:
    uploaded_backup = st.file_uploader("导入备份CSV恢复", type=["csv"], label_visibility="collapsed")
    if uploaded_backup is not None and st.button("📥 恢复"):
        raw = uploaded_backup.getvalue()
        df_imp = None
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                df_imp = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                df_imp = None
        if df_imp is None:
            st.sidebar.error("恢复失败：CSV读取失败（建议UTF-8编码）。")
        else:
            df_imp = normalize_df(df_imp)
            # 补全必须字段
            if "fetched_at" not in df_imp.columns:
                df_imp["fetched_at"] = pd.Timestamp.now()
            # 写库
            df_imp["pubdate"] = pd.to_datetime(df_imp["pubdate"], errors="coerce")
            df_imp["fetched_at"] = pd.to_datetime(df_imp["fetched_at"], errors="coerce").fillna(pd.Timestamp.now())
            # 转成字符串入库
            df_imp["pubdate"] = df_imp["pubdate"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_imp["fetched_at"] = df_imp["fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_imp)
            st.sidebar.success("恢复完成（已写入数据库）。")

with st.sidebar.expander("危险操作：清空全部数据", expanded=False):
    if st.button("🗑️ 清空数据库（不可撤销）"):
        clear_all_data()
        st.sidebar.success("已清空。刷新页面即可看到空数据。")

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
            # stringify datetime for DB
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（数据已保存）")

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

                # stringify datetime for DB
                df_csv["pubdate"] = pd.to_datetime(df_csv["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_csv["fetched_at"] = pd.to_datetime(df_csv["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")

                upsert_rows(df_csv)
                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（数据已保存）")

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
# Project filter: hide baseline project
# =========================
projects = sorted([p for p in df_db["project"].dropna().unique().tolist() if str(p).strip() != "" and p != BASELINE_PROJECT])
sel_projects = st.sidebar.multiselect("选择项目（筛选展示）", projects, default=projects if projects else None)

df_main = df_db[df_db["project"] != BASELINE_PROJECT].copy()
df_f = df_main[df_main["project"].isin(sel_projects)].copy() if sel_projects else df_main.copy()

# =========================
# KPI cards
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

# =========================
# KOL module (independent, uses baseline saved in DB)
# =========================
st.divider()
st.subheader("KOL合作资料库（独立模块：数据也会保存，可跨设备查看）")

all_projects = projects
default_collab = sel_projects if sel_projects else all_projects

with st.expander("KOL模块设置（默认即可）", expanded=False):
    collab_projects = st.multiselect("哪些项目算“合作项目”", all_projects, default=default_collab)
    baseline_pref = st.radio(
        "日常基准怎么取？",
        ["优先用非合作项目视频（更像‘日常’）", "用该UP在库里所有视频（更宽松）"],
        index=0
    )
    min_baseline_n = st.slider("基准最少需要多少条视频（太少不判定）", 3, 30, 6)
    extra_baseline_n = st.slider("自动补齐：每个KOL额外抓几条日常视频", 0, 10, 5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)

    lift_view_pct = st.slider("播放提升阈值（相对基准中位数）", 0, 300, 30, step=5)
    lift_er_pct = st.slider("互动率提升阈值（相对基准中位数）", 0, 300, 20, step=5)
    lift_deep_pct = st.slider("深度信号提升阈值（相对基准中位数）", 0, 300, 10, step=5)
    z_threshold = st.slider("Z分数阈值", 0.0, 3.0, 1.0, step=0.1)
    require_both = st.checkbox("更严格：播放&互动率都要明显更好才标注", value=False)

cA, cB, cC = st.columns([1,1,2])
with cA:
    fetch_baseline_btn = st.button("🧲 自动抓KOL日常样本（补齐4-5条，保存）")
with cB:
    build_kol_btn = st.button("📚 生成/刷新KOL商务资料库")
with cC:
    st.caption("流程：选合作项目 →（可选）补齐日常样本 → 生成资料库 → 下载CSV做合作池")

if collab_projects:
    collab_df = df_db[df_db["project"].isin(collab_projects)].copy()

    if fetch_baseline_btn:
        if collab_df.empty or extra_baseline_n <= 0:
            st.warning("合作项目下没有数据，或补齐数量为0。")
        else:
            existed = set(df_db["bvid"].astype(str).tolist())
            rows_new = []

            for up, g in collab_df.groupby("owner_name"):
                mids = g["owner_mid"].dropna().unique().tolist()
                if not mids:
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

            if rows_new:
                df_new = normalize_df(pd.DataFrame(rows_new))
                df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                upsert_rows(df_new)
                st.success(f"已保存KOL日常样本：新增 {len(rows_new)} 条（不会出现在项目归档）")
                st.rerun()
            else:
                st.warning("未抓到可新增的日常样本（可能限流/接口波动/样本已存在）。")

    if build_kol_btn:
        df_all = df_db.copy()
        df_all = compute_metrics(df_all)

        collab_df2 = df_all[df_all["project"].isin(collab_projects)].copy()
        if collab_df2.empty:
            st.warning("合作项目下没有数据，无法生成资料库。")
        else:
            rows = []
            for up, g_collab in collab_df2.groupby("owner_name"):
                base_hidden = df_all[(df_all["project"] == BASELINE_PROJECT) & (df_all["baseline_for"] == up)].copy()

                if baseline_pref.startswith("优先用非合作"):
                    base_non_collab = df_all[
                        (df_all["owner_name"] == up)
                        & (~df_all["project"].isin(collab_projects))
                        & (df_all["project"] != BASELINE_PROJECT)
                    ].copy()
                    g_base = pd.concat([base_hidden, base_non_collab], ignore_index=True)
                    if len(g_base) < min_baseline_n:
                        g_base = pd.concat([base_hidden, df_all[df_all["owner_name"] == up]], ignore_index=True)
                else:
                    g_base = pd.concat([base_hidden, df_all[df_all["owner_name"] == up]], ignore_index=True)

                g_base = g_base.drop_duplicates(subset=["bvid"], keep="last")
                if len(g_base) < min_baseline_n:
                    continue

                base_view_med = float(g_base["view"].median())
                base_er_med = float(g_base["engagement_rate"].median())
                base_deep_med = float(g_base["deep_signal_ratio"].median())

                base_view_mean = float(g_base["view"].mean())
                base_view_std = float(g_base["view"].std(ddof=0)) if float(g_base["view"].std(ddof=0)) > 1e-9 else 0.0
                base_er_mean = float(g_base["engagement_rate"].mean())
                base_er_std = float(g_base["engagement_rate"].std(ddof=0)) if float(g_base["engagement_rate"].std(ddof=0)) > 1e-12 else 0.0
                base_deep_mean = float(g_base["deep_signal_ratio"].mean())
                base_deep_std = float(g_base["deep_signal_ratio"].std(ddof=0)) if float(g_base["deep_signal_ratio"].std(ddof=0)) > 1e-12 else 0.0

                collab_view_med = float(g_collab["view"].median())
                collab_er_med = float(g_collab["engagement_rate"].median())
                collab_deep_med = float(g_collab["deep_signal_ratio"].median())

                view_lift = (collab_view_med / base_view_med - 1.0) if base_view_med > 0 else np.nan
                er_lift = (collab_er_med / base_er_med - 1.0) if base_er_med > 0 else np.nan
                deep_lift = (collab_deep_med / base_deep_med - 1.0) if base_deep_med > 0 else np.nan

                z_view = (collab_view_med - base_view_mean) / base_view_std if base_view_std > 0 else 0.0
                z_er = (collab_er_med - base_er_mean) / base_er_std if base_er_std > 0 else 0.0
                z_deep = (collab_deep_med - base_deep_mean) / base_deep_std if base_deep_std > 0 else 0.0

                cond_view = (not np.isnan(view_lift)) and (view_lift >= lift_view_pct/100.0) and (z_view >= z_threshold)
                cond_er = (not np.isnan(er_lift)) and (er_lift >= lift_er_pct/100.0) and (z_er >= z_threshold)
                cond_deep = (not np.isnan(deep_lift)) and (deep_lift >= lift_deep_pct/100.0) and (z_deep >= z_threshold)

                is_good = (cond_view and cond_er) if require_both else (cond_view or cond_er or cond_deep)

                top3 = g_collab.sort_values("view", ascending=False).head(3)
                top3_titles = "｜".join([str(t)[:30] for t in top3["title"].tolist()])
                top3_links = "｜".join([f"https://www.bilibili.com/video/{b}" for b in top3["bvid"].tolist()])

                # tags
                collab_sorted = g_collab.sort_values("view", ascending=False)
                total_view = float(collab_sorted["view"].sum())
                top1_share = (float(collab_sorted.iloc[0]["view"]) / total_view) if total_view > 0 else 0.0
                er_q1 = float(g_collab["engagement_rate"].quantile(0.25))
                er_q3 = float(g_collab["engagement_rate"].quantile(0.75))
                er_iqr = er_q3 - er_q1

                tags = []
                if not np.isnan(view_lift) and view_lift >= 0.30: tags.append("热度拉升型")
                if not np.isnan(er_lift) and er_lift >= 0.20: tags.append("强互动引爆")
                if not np.isnan(deep_lift) and deep_lift >= 0.10: tags.append("价值沉淀型")
                if top1_share >= 0.55: tags.append("头部依赖高")
                if er_iqr >= float(g_collab["engagement_rate"].median()) * 0.8: tags.append("波动较大")
                if not tags: tags.append("常规表现")

                heat = "热度拉升" if (not np.isnan(view_lift) and view_lift >= 0.30) else "热度稳定"
                interact = "互动强" if (not np.isnan(er_lift) and er_lift >= 0.20) else "互动常规"
                depth = "沉淀强" if (not np.isnan(deep_lift) and deep_lift >= 0.10) else "沉淀一般"
                risk = []
                if "头部依赖高" in tags: risk.append("头部依赖")
                if "波动较大" in tags: risk.append("波动")
                persona = f"{heat} + {interact} + {depth}" + (f"（风险：{'/'.join(risk)}）" if risk else "")

                if (not np.isnan(view_lift) and view_lift >= 0.30) and (not np.isnan(er_lift) and er_lift >= 0.20):
                    scene = "大促节点/新品首发/热点借势"
                    form = "首发测评/挑战赛/联合企划（带话题）"
                    hook = "前3秒强卖点 + 互动任务（提问/投票） + 结尾投币收藏理由"
                    avoid = "避免硬广直给，必须故事化/体验化"
                elif (not np.isnan(deep_lift) and deep_lift >= 0.10):
                    scene = "口碑向/种草向/长尾持续曝光"
                    form = "系列化栏目/深度测评/清单向内容"
                    hook = "可复看价值点 + 收藏引导 + 评论区置顶资料"
                    avoid = "别用纯播放KPI考核；重点看收藏/投币/长尾"
                else:
                    scene = "低成本试水/补位投放"
                    form = "单条软植入/素材共创/话题互动"
                    hook = "围绕TA擅长结构（整活/测评/盘点）做轻合作"
                    avoid = "不要重权益绑定，先1-2条验证再加码"

                if "头部依赖高" in tags: avoid += "；建议AB备选选题"
                if "波动较大" in tags: avoid += "；建议明确brief与资源位支持"
                suggestion_bundle = f"适合场景：{scene}｜合作形式：{form}｜内容抓手：{hook}｜避坑：{avoid}"

                if is_good and ("头部依赖高" not in tags) and ("波动较大" not in tags):
                    advice = "优先续约/可加码：合作有明确增益，可争取更深权益/系列化"
                elif is_good:
                    advice = "可合作但控风险：建议AB选题+加强分发+明确转化KPI"
                elif (not is_good) and (not np.isnan(deep_lift) and deep_lift >= 0.10):
                    advice = "小众高质：适合垂类/口碑场景，不建议用纯播放KPI"
                else:
                    advice = "谨慎：先低成本试水或换选题/包装后再评估"

                rows.append({
                    "KOL/UP主": up,
                    "标注": "⭐ 合作明显更好" if is_good else "",
                    "标签": "、".join(tags),
                    "KOL画像一句话": persona,
                    "合作建议组合": suggestion_bundle,
                    "商务建议": advice,
                    "合作视频数": int(len(g_collab)),
                    "基准视频数": int(len(g_base)),
                    "合作播放中位数": collab_view_med,
                    "日常播放中位数": base_view_med,
                    "合作播放提升": view_lift,
                    "合作互动率中位数": collab_er_med,
                    "日常互动率中位数": base_er_med,
                    "合作互动率提升": er_lift,
                    "合作深度信号中位数": collab_deep_med,
                    "日常深度信号中位数": base_deep_med,
                    "深度信号提升": deep_lift,
                    "证据-合作Top3标题": top3_titles,
                    "证据-合作Top3链接": top3_links,
                })

            if not rows:
                st.warning("KOL资料库生成失败：日常基准视频不足。建议先点“自动抓KOL日常样本”。")
            else:
                lib = pd.DataFrame(rows)
                lib["_flag"] = lib["标注"].apply(lambda x: 1 if str(x).strip() else 0)
                lib = lib.sort_values(["_flag","合作互动率提升","合作播放提升"], ascending=[False, False, False]).drop(columns=["_flag"])

                st.dataframe(lib, use_container_width=True, height=420)
                st.download_button(
                    "⬇️ 下载KOL商务资料库（CSV）",
                    data=lib.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kol_business_library.csv",
                    mime="text/csv"
                )

# =========================
# Project table
# =========================
st.divider()
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project","bvid","title","owner_name","pubdate",
    "view","like","coin","favorite","reply","engagement_rate","deep_signal_ratio"
]
st.dataframe(df_f[show_cols].sort_values("view", ascending=False), use_container_width=True, height=320)

# =========================
# Top/Bottom
# =========================
st.subheader("Top / Bottom 深挖（含Up主基准对比）")
for proj in sel_projects if sel_projects else projects:
    d = df_f[df_f["project"] == proj].sort_values("view", ascending=False)
    if d.empty:
        continue
    top = d.iloc[0]
    bottom = d.iloc[-1]

    st.markdown(f"### 项目：{proj}")
    left, right = st.columns(2)

    def render_card(col, row, tag):
        up = row["owner_name"]
        base = df_f[df_f["owner_name"] == up]
        mean_v, std_v = base["view"].mean(), base["view"].std(ddof=0)
        mean_er, std_er = base["engagement_rate"].mean(), base["engagement_rate"].std(ddof=0)

        col.markdown(f"**{tag}：{row['title']}**")
        col.caption(f"UP：{up} ｜ BV：{row['bvid']} ｜ 发布：{row['pubdate']}")
        col.metric("播放", f"{int(row['view']):,}", label_vs_baseline(row["view"], mean_v, std_v))
        col.metric("互动率", f"{row['engagement_rate']*100:.2f}%", label_vs_baseline(row["engagement_rate"], mean_er, std_er))
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
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放），互动率 {best['engagement_rate']*100:.2f}%，深度信号占比 {best['deep_signal_ratio']*100:.1f}%。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放），互动率 {worst['engagement_rate']*100:.2f}%。建议检查封面/标题信息密度与投放时段，并在评论区做更强的互动引导。"
)
if df_f["deep_signal_ratio"].mean() < 0.35:
    insights.append("3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”，建议强化：价值点前置、结尾引导收藏/投币、增加系列化承诺。")
else:
    insights.append("3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性，可考虑围绕该方向做系列化与固定栏目节奏。")
st.write("\n".join(insights))
