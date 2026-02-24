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

def _pct(x: float, digits=1):
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "-"

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
            if low in ["project","url","bvid","title","owner_name","owner_mid","pubdate",
                       "view","like","coin","favorite","reply","danmaku","share","fans_delta","fetched_at",
                       "baseline_for","data_type"]:
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
    else:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df["fetched_at"] = df["fetched_at"].fillna(pd.Timestamp.now())

    keep = set([
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta","fetched_at",
        "baseline_for","data_type"
    ])
    existing = [c for c in df.columns if c in keep]
    df = df[existing].copy()

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

def fetch_recent_bvids_by_mid(mid: int, n: int = 5) -> list[dict]:
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
        if not bvid:
            continue
        out.append({
            "bvid": bvid,
            "title": v.get("title", ""),
            "pubdate": pd.to_datetime(v.get("created", 0), unit="s", errors="coerce"),
        })
    return out

# =========================
# Session State
# =========================
if "rows" not in st.session_state:
    st.session_state["rows"] = []

def append_df_to_session(df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return
    existing = pd.DataFrame(st.session_state["rows"])
    if existing.empty:
        st.session_state["rows"] = normalize_df(df_new).to_dict("records")
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
    "baseline_for": "",
    "data_type": "collab",
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
                row["data_type"] = "collab"
                rows.append(row)
                ok += 1
                time.sleep(0.4)
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
                if "data_type" not in df_csv.columns:
                    df_csv["data_type"] = "collab"
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

st.divider()

# =========================================================
# ✅ 商务KOL资料库：画像一句话 + 建议组合 + 自动补齐日常样本
# =========================================================
st.subheader("KOL合作资料库（商务版：标注 + 标签 + 画像一句话 + 合作建议组合 + 可下载）")

all_projects = sorted([p for p in df["project"].dropna().unique().tolist() if str(p).strip() != ""])
default_collab = sel_projects if sel_projects else all_projects

with st.expander("设置（建议保持默认，先跑起来）", expanded=False):
    collab_projects = st.multiselect("哪些项目算“合作项目”", all_projects, default=default_collab)
    baseline_pref = st.radio(
        "基准优先取什么？",
        ["优先用非合作项目视频（更像‘日常’）", "用该UP在库里所有视频（更宽松）"],
        index=0
    )
    min_baseline_n = st.slider("基准最少需要多少条视频", 3, 30, 6)
    extra_baseline_n = st.slider("自动补齐：每个KOL额外抓几条日常视频", 0, 10, 5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.6, step=0.1)

    # “明显更好”判定
    lift_view_pct = st.slider("播放提升阈值（相对基准中位数）", 0, 300, 30, step=5)
    lift_er_pct = st.slider("互动率提升阈值（相对基准中位数）", 0, 300, 20, step=5)
    lift_deep_pct = st.slider("深度信号提升阈值（相对基准中位数）", 0, 300, 10, step=5)
    z_threshold = st.slider("Z分数阈值", 0.0, 3.0, 1.0, step=0.1)
    require_both = st.checkbox("更严格：播放&互动率都要明显更好才标注", value=False)

colA, colB, colC = st.columns([1,1,2])
with colA:
    btn_fetch_baseline = st.button("🧲 自动抓KOL日常样本（补齐对比）")
with colB:
    btn_build_library = st.button("📚 生成/刷新KOL资料库")
with colC:
    st.caption("用法：选合作项目 →（可选）补齐日常样本 → 生成资料库 → 下载CSV作为合作池底稿")

if not collab_projects:
    st.info("请先选择至少一个“合作项目”。")
else:
    df_all = compute_metrics(normalize_df(pd.DataFrame(st.session_state["rows"])))
    df_all = df_all[df_all["owner_name"].astype(str).str.strip() != ""].copy()
    collab_df = df_all[df_all["project"].isin(collab_projects)].copy()

    # ---------- 自动补齐样本 ----------
    if btn_fetch_baseline:
        if collab_df.empty or extra_baseline_n <= 0:
            st.warning("合作项目无数据，或补齐数量为0。")
        else:
            rows_new = []
            existed = set(df_all["bvid"].astype(str).tolist())

            for up, g in collab_df.groupby("owner_name"):
                mid_vals = g["owner_mid"].dropna().unique().tolist() if "owner_mid" in g.columns else []
                if not mid_vals:
                    continue
                mid = mid_vals[0]
                try:
                    recent = fetch_recent_bvids_by_mid(int(mid), n=extra_baseline_n)
                except Exception:
                    continue

                for item in recent:
                    bvid = item["bvid"]
                    if bvid in existed:
                        continue
                    try:
                        row = fetch_video_stats_by_bvid(bvid)
                        row["project"] = f"BASELINE::{up}"
                        row["baseline_for"] = up
                        row["data_type"] = "baseline"
                        row["url"] = f"https://www.bilibili.com/video/{bvid}"
                        rows_new.append(row)
                        existed.add(bvid)
                        time.sleep(float(sleep_sec))
                    except Exception:
                        continue

            if rows_new:
                append_df_to_session(pd.DataFrame(rows_new))
                st.success(f"已补齐日常样本：新增 {len(rows_new)} 条（项目名以 BASELINE::UP主 标识）")
            else:
                st.warning("未抓到可新增的日常样本（可能限流/接口波动/样本已在库中）。")

        df_all = compute_metrics(normalize_df(pd.DataFrame(st.session_state["rows"])))
        df_all = df_all[df_all["owner_name"].astype(str).str.strip() != ""].copy()
        collab_df = df_all[df_all["project"].isin(collab_projects)].copy()

    # ---------- 生成资料库 ----------
    if btn_build_library:
        if collab_df.empty:
            st.warning("合作项目下没有数据，无法生成资料库。")
        else:
            rows = []
            for up, g_collab in collab_df.groupby("owner_name"):
                # baseline
                if baseline_pref.startswith("优先用非合作"):
                    g_base = df_all[(df_all["owner_name"] == up) & (~df_all["project"].isin(collab_projects))].copy()
                    if len(g_base) < min_baseline_n:
                        g_base = df_all[df_all["owner_name"] == up].copy()
                else:
                    g_base = df_all[df_all["owner_name"] == up].copy()

                if len(g_base) < min_baseline_n:
                    continue

                # baseline stats
                base_view_med = float(g_base["view"].median())
                base_er_med = float(g_base["engagement_rate"].median())
                base_deep_med = float(g_base["deep_signal_ratio"].median())

                base_view_mean = float(g_base["view"].mean())
                base_view_std = float(g_base["view"].std(ddof=0)) if float(g_base["view"].std(ddof=0)) > 1e-9 else 0.0
                base_er_mean = float(g_base["engagement_rate"].mean())
                base_er_std = float(g_base["engagement_rate"].std(ddof=0)) if float(g_base["engagement_rate"].std(ddof=0)) > 1e-12 else 0.0
                base_deep_mean = float(g_base["deep_signal_ratio"].mean())
                base_deep_std = float(g_base["deep_signal_ratio"].std(ddof=0)) if float(g_base["deep_signal_ratio"].std(ddof=0)) > 1e-12 else 0.0

                # collab stats
                collab_view_med = float(g_collab["view"].median())
                collab_er_med = float(g_collab["engagement_rate"].median())
                collab_deep_med = float(g_collab["deep_signal_ratio"].median())

                # lifts
                view_lift = (collab_view_med / base_view_med - 1.0) if base_view_med > 0 else np.nan
                er_lift = (collab_er_med / base_er_med - 1.0) if base_er_med > 0 else np.nan
                deep_lift = (collab_deep_med / base_deep_med - 1.0) if base_deep_med > 0 else np.nan

                # z
                z_view = (collab_view_med - base_view_mean) / base_view_std if base_view_std > 0 else 0.0
                z_er = (collab_er_med - base_er_mean) / base_er_std if base_er_std > 0 else 0.0
                z_deep = (collab_deep_med - base_deep_mean) / base_deep_std if base_deep_std > 0 else 0.0

                cond_view = (not np.isnan(view_lift)) and (view_lift >= lift_view_pct/100.0) and (z_view >= z_threshold)
                cond_er = (not np.isnan(er_lift)) and (er_lift >= lift_er_pct/100.0) and (z_er >= z_threshold)
                cond_deep = (not np.isnan(deep_lift)) and (deep_lift >= lift_deep_pct/100.0) and (z_deep >= z_threshold)

                if require_both:
                    is_good = cond_view and cond_er
                else:
                    is_good = cond_view or cond_er or cond_deep

                # evidence
                top3 = g_collab.sort_values("view", ascending=False).head(3)
                top3_titles = "｜".join([str(t)[:30] for t in top3["title"].tolist()])
                top3_links = "｜".join([f"https://www.bilibili.com/video/{b}" for b in top3["bvid"].tolist()])

                # stability/head dependence inside collab
                collab_sorted = g_collab.sort_values("view", ascending=False)
                total_view = float(collab_sorted["view"].sum())
                top1_share = (float(collab_sorted.iloc[0]["view"]) / total_view) if total_view > 0 else 0.0

                er_q1 = float(g_collab["engagement_rate"].quantile(0.25))
                er_q3 = float(g_collab["engagement_rate"].quantile(0.75))
                er_iqr = er_q3 - er_q1

                # ===== Business tags =====
                tags = []
                if not np.isnan(view_lift) and view_lift >= 0.30:
                    tags.append("热度拉升型")
                if not np.isnan(deep_lift) and deep_lift >= 0.10:
                    tags.append("价值沉淀型")
                if not np.isnan(er_lift) and er_lift >= 0.20:
                    tags.append("强互动引爆")
                if top1_share >= 0.55:
                    tags.append("头部依赖高")
                if er_iqr >= float(g_collab["engagement_rate"].median()) * 0.8:
                    tags.append("波动较大")
                if not tags:
                    tags.append("常规表现")

                # ===== One-line persona =====
                # 核心判断：热度(播放提升) / 互动(互动率提升) / 沉淀(深度信号提升) / 稳定性 / 头部依赖
                heat = "热度拉升" if (not np.isnan(view_lift) and view_lift >= 0.30) else "热度稳定"
                interact = "互动强" if (not np.isnan(er_lift) and er_lift >= 0.20) else "互动常规"
                depth = "沉淀强" if (not np.isnan(deep_lift) and deep_lift >= 0.10) else "沉淀一般"
                risk = []
                if "头部依赖高" in tags:
                    risk.append("头部依赖")
                if "波动较大" in tags:
                    risk.append("波动")
                risk_txt = f"（风险：{'/'.join(risk)}）" if risk else ""
                persona = f"{heat} + {interact} + {depth}{risk_txt}"

                # ===== Cooperation suggestion bundle =====
                # 适合场景/合作形式/内容抓手/避坑点
                if (not np.isnan(view_lift) and view_lift >= 0.30) and (not np.isnan(er_lift) and er_lift >= 0.20):
                    scene = "大促节点/新品首发/热点借势"
                    form = "首发测评/挑战赛/联合企划（带话题）"
                    hook = "前3秒强卖点 + 明确互动任务（评论提问/投票） + 结尾投币收藏理由"
                    avoid = "避免硬广直给，必须故事化/体验化"
                elif (not np.isnan(deep_lift) and deep_lift >= 0.10):
                    scene = "口碑向/种草向/长尾持续曝光"
                    form = "系列化栏目/深度测评/清单向内容"
                    hook = "可复看价值点（技巧/攻略/对比）+ 收藏引导 + 评论区置顶资料"
                    avoid = "别用纯播放KPI考核；重点看收藏/投币/搜索长尾"
                else:
                    scene = "低成本试水/补位投放"
                    form = "单条软植入/素材共创/话题互动"
                    hook = "围绕TA擅长的内容结构（整活/测评/盘点）做轻合作"
                    avoid = "不要重权益绑定，先用1-2条验证再加码"

                if "头部依赖高" in tags:
                    avoid += "；同时建议准备AB两条备选选题，降低单条失败风险"
                if "波动较大" in tags:
                    avoid += "；建议给明确brief与资源位支持，减少发挥波动"

                suggestion_bundle = f"适合场景：{scene}｜合作形式：{form}｜内容抓手：{hook}｜避坑：{avoid}"

                # ===== Business advice (short) =====
                if is_good and ("头部依赖高" not in tags) and ("波动较大" not in tags):
                    advice = "优先续约/可加码：合作对其账号表现有明确增益，可争取更深权益/系列化"
                elif is_good and ("头部依赖高" in tags or "波动较大" in tags):
                    advice = "可合作但要控风险：建议做AB选题+加强分发资源+明确转化KPI"
                elif (not is_good) and (not np.isnan(deep_lift) and deep_lift >= 0.10):
                    advice = "小众高质：适合垂类/口碑场景，不建议用纯播放KPI评估"
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

                    "头部依赖(Top1贡献)": top1_share,
                    "互动率波动(IQR)": er_iqr,

                    "证据-合作Top3标题": top3_titles,
                    "证据-合作Top3链接": top3_links,
                })

            if not rows:
                st.warning("数据不足：缺少KOL日常基准视频。建议先点“自动抓KOL日常样本”。")
            else:
                lib = pd.DataFrame(rows)
                lib["_flag"] = lib["标注"].apply(lambda x: 1 if str(x).strip() else 0)
                lib = lib.sort_values(["_flag","合作互动率提升","合作播放提升"], ascending=[False, False, False]).drop(columns=["_flag"])

                # 展示（格式化）
                show = lib.copy()
                show["合作播放中位数"] = show["合作播放中位数"].map(lambda x: f"{int(x):,}")
                show["日常播放中位数"] = show["日常播放中位数"].map(lambda x: f"{int(x):,}")
                show["合作播放提升"] = show["合作播放提升"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")

                show["合作互动率中位数"] = show["合作互动率中位数"].map(lambda x: f"{x*100:.2f}%")
                show["日常互动率中位数"] = show["日常互动率中位数"].map(lambda x: f"{x*100:.2f}%")
                show["合作互动率提升"] = show["合作互动率提升"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")

                show["合作深度信号中位数"] = show["合作深度信号中位数"].map(lambda x: f"{x*100:.1f}%")
                show["日常深度信号中位数"] = show["日常深度信号中位数"].map(lambda x: f"{x*100:.1f}%")
                show["深度信号提升"] = show["深度信号提升"].map(lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")

                show["头部依赖(Top1贡献)"] = show["头部依赖(Top1贡献)"].map(lambda x: f"{x*100:.1f}%")
                show["互动率波动(IQR)"] = show["互动率波动(IQR)"].map(lambda x: f"{x*100:.2f}pp")

                st.markdown("**KOL资料库（商务版）**")
                st.dataframe(show, use_container_width=True, height=440)

                out_bytes = lib.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ 下载KOL商务资料库（CSV）", data=out_bytes, file_name="kol_business_library.csv", mime="text/csv")

                flagged = lib[lib["标注"].astype(str).str.contains("⭐")]
                if not flagged.empty:
                    st.success(f"已标注 {len(flagged)} 位“合作明显更好”的KOL（建议优先作为未来合作池）。")
                else:
                    st.warning("当前阈值下暂无“合作明显更好”的KOL。可适当降低提升阈值或Z阈值。")

st.divider()

# =========================
# 视频表（含baseline标记，便于你核对）
# =========================
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project","data_type","baseline_for",
    "bvid","title","owner_name","pubdate",
    "view","like","coin","favorite","reply","danmaku","share","fans_delta",
    "engagement_rate","deep_signal_ratio"
]
existing_cols = [c for c in show_cols if c in df_f.columns]
st.dataframe(
    df_f[existing_cols].sort_values("view", ascending=False),
    use_container_width=True,
    height=360,
)

# =========================
# 互动率分布
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
