import os
import re
import time
import io
import sqlite3
import hashlib
import urllib.parse
import json
import random
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

# ✅ DB固定到 app.py 同目录（避免工作目录变化导致“新建空库→基准全没”）
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "bili_dashboard.db")

# ✅ 自动备份（尽可能减少“每周打开空库”）
BACKUP_DIR = os.path.join(APP_DIR, "backup")
BACKUP_LATEST_CSV = os.path.join(BACKUP_DIR, "backup_latest.csv")

TABLE_NAME = "videos"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://space.bilibili.com",
    "Referer": "https://space.bilibili.com/",
}

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

def _ensure_backup_dir():
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
    except Exception:
        pass

def _save_backup_csv(df_all: pd.DataFrame):
    """
    将全量库写到 backup_latest.csv
    """
    if df_all is None or df_all.empty:
        return
    _ensure_backup_dir()
    try:
        df_all.to_csv(BACKUP_LATEST_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        # 备份失败不要影响主流程
        pass

def _try_restore_from_backup() -> bool:
    """
    如果DB为空，尝试从 backup_latest.csv 恢复到DB
    返回是否恢复成功
    """
    if not os.path.exists(BACKUP_LATEST_CSV):
        return False
    try:
        raw = open(BACKUP_LATEST_CSV, "rb").read()
    except Exception:
        return False

    df_imp = None
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            df_imp = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:
            df_imp = None

    if df_imp is None or df_imp.empty:
        return False

    df_imp = normalize_df(df_imp)
    if "fetched_at" not in df_imp.columns:
        df_imp["fetched_at"] = pd.Timestamp.now()

    df_imp["pubdate"] = pd.to_datetime(df_imp["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df_imp["fetched_at"] = pd.to_datetime(df_imp["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")

    upsert_rows(df_imp, skip_backup=True)
    return True

def load_all_rows() -> pd.DataFrame:
    """
    读取DB；如果为空，自动尝试从 backup_latest.csv 恢复一次再读
    """
    init_db()
    with db_conn() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    if df is not None and not df.empty:
        return df

    # DB为空 -> 尝试恢复
    restored = _try_restore_from_backup()
    if restored:
        with db_conn() as conn:
            df2 = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        return df2

    return df

def upsert_rows(df_new: pd.DataFrame, skip_backup: bool = False):
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

    # ✅ 每次写入后自动备份（尽可能防止“打开空库”）
    if not skip_backup:
        try:
            df_all = load_all_rows()
            _save_backup_csv(df_all)
        except Exception:
            pass

def clear_all_data():
    init_db()
    with db_conn() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME}")
        conn.commit()
    # 不删除备份：避免误点“清空”后无处恢复

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

def _norm_mid(x) -> str:
    """mid 统一为纯数字字符串；超长mid视为异常。"""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return ""
    if len(s) > 12:
        return ""
    return s

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
        "owner_mid": "owner_mid",
        "mid": "owner_mid",
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

    if "owner_mid" not in df.columns:
        df["owner_mid"] = ""
    df["owner_mid"] = df["owner_mid"].apply(_norm_mid)

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

def _sort_owner_hist(df_owner: pd.DataFrame) -> pd.DataFrame:
    g = df_owner.copy()
    g["__sort_time"] = g["pubdate"]
    missing = g["__sort_time"].isna()
    g.loc[missing, "__sort_time"] = g.loc[missing, "fetched_at"]
    g = g[pd.notna(g["__sort_time"])].sort_values("__sort_time", ascending=False)
    return g

# =========================
# Performance labels (用于Top/Bottom和表格的“发挥”标签)
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
        cache[up] = _sort_owner_hist(g)
    return cache

def recent_baseline(owner_hist_desc: pd.DataFrame, current_bvid: str, col: str, window_n: int) -> np.ndarray:
    if owner_hist_desc is None or owner_hist_desc.empty:
        return np.array([], dtype=float)
    h = owner_hist_desc[owner_hist_desc["bvid"] != current_bvid]
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
# ✅ WBI 签名
# =========================
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32,
    15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19,
    29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61,
    26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63,
    57, 62, 11, 36, 20, 34, 44, 52,
]

def _get_mixin_key(img_key: str, sub_key: str) -> str:
    s = img_key + sub_key
    return "".join([s[i] for i in _MIXIN_KEY_ENC_TAB])[:32]

@st.cache_data(ttl=60*30)
def _get_wbi_keys() -> tuple[str, str]:
    nav = "https://api.bilibili.com/x/web-interface/nav"
    r = requests.get(nav, headers=HEADERS, timeout=10)
    j = r.json()
    wbi_img = (j.get("data") or {}).get("wbi_img") or {}
    img_url = wbi_img.get("img_url", "")
    sub_url = wbi_img.get("sub_url", "")
    img_key = img_url.split("/")[-1].split(".")[0]
    sub_key = sub_url.split("/")[-1].split(".")[0]
    return img_key, sub_key

def _wbi_sign(params: dict) -> dict:
    img_key, sub_key = _get_wbi_keys()
    mixin_key = _get_mixin_key(img_key, sub_key)

    params = {k: v for k, v in params.items() if v is not None}
    params["wts"] = int(time.time())

    def _filter(v):
        return re.sub(r"[!'()*]", "", str(v))

    sorted_items = sorted((k, _filter(v)) for k, v in params.items())
    query = urllib.parse.urlencode(sorted_items)
    w_rid = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    params["w_rid"] = w_rid
    return params

# =========================
# B站抓取
# =========================
def _sleep_jitter(base: float = 0.6):
    """轻微随机等待，降低连续请求特征。"""
    try:
        base = float(base)
    except Exception:
        base = 0.6
    time.sleep(max(0.1, base) + random.uniform(0.05, 0.35))


def _make_bili_session(referer: str = "https://www.bilibili.com/") -> requests.Session:
    """
    为每次抓取建立带常见浏览器头与基础cookie的 Session。
    目标不是绕过风控，而是让请求上下文更接近真实浏览器访问，减少接口偶发空列表。
    """
    sess = requests.Session()
    h = HEADERS.copy()
    h.update({
        "Referer": referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    })
    sess.headers.update(h)
    try:
        sess.get("https://www.bilibili.com/", timeout=8)
    except Exception:
        pass
    return sess


def _as_video_row(v: dict) -> dict | None:
    """兼容 B站不同接口/页面里常见的视频字段。"""
    if not isinstance(v, dict):
        return None
    bvid = v.get("bvid") or v.get("bvidStr") or v.get("BV")
    if not bvid:
        uri = _safe_str(v.get("uri") or v.get("arcurl") or v.get("url") or "")
        bvid = parse_bvid(uri)
    if not bvid:
        return None

    created = v.get("created") or v.get("pubdate") or v.get("ctime")
    if created is not None and _safe_int(created) < 1000000000:
        created = None

    stat = v.get("stat") or {}
    return {
        "bvid": bvid,
        "title": v.get("title", ""),
        "pubdate": pd.to_datetime(created or 0, unit="s", errors="coerce"),
        "view": _safe_int(v.get("play", stat.get("view", v.get("view", 0)))),
        "reply": _safe_int(v.get("comment", stat.get("reply", v.get("reply", 0)))),
    }


def _dedupe_rows(rows: list[dict], n: int) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        bvid = r.get("bvid") if isinstance(r, dict) else None
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        out.append(r)
        if len(out) >= n:
            break
    return out


def _extract_video_rows_from_html(html: str, n: int = 30) -> list[dict]:
    """从UP空间页源码里兜底提取 BV 号，再用详情接口补数据。"""
    if not html:
        return []

    bvids = []
    for bv in re.findall(r"BV[0-9A-Za-z]{10}", html):
        if bv not in bvids:
            bvids.append(bv)
        if len(bvids) >= n:
            break

    rows = []
    for bvid in bvids[:n]:
        detail = fetch_video_detail_by_bvid(bvid)
        if detail is not None:
            rows.append({
                "bvid": bvid,
                "title": detail.get("title", ""),
                "pubdate": detail.get("pubdate", pd.NaT),
                "view": _safe_int(detail.get("view", 0)),
                "reply": _safe_int(detail.get("reply", 0)),
            })
        else:
            rows.append({"bvid": bvid, "title": "", "pubdate": pd.NaT, "view": 0, "reply": 0})
        _sleep_jitter(0.25)
    return rows


def _open_space_with_headless_browser(mid: int, wait_sec: float = 4.0) -> tuple[str, dict]:
    """
    可选兜底：如果本机/部署环境装了 selenium + chrome，就无头打开UP主页。
    没有依赖时静默返回空，不影响原有功能。
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception:
        return "", {}

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1366,900")
        options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
        driver = webdriver.Chrome(options=options)
        url = f"https://space.bilibili.com/{mid}/video"
        driver.get(url)
        time.sleep(max(2.5, float(wait_sec)))
        html = driver.page_source or ""
        cookies = {c.get("name"): c.get("value") for c in driver.get_cookies() if c.get("name")}
        return html, cookies
    except Exception:
        return "", {}
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


def fetch_video_detail_by_bvid(bvid: str) -> dict | None:
    api = "https://api.bilibili.com/x/web-interface/view"
    sess = _make_bili_session(referer=f"https://www.bilibili.com/video/{bvid}")
    for _ in range(3):
        try:
            r = sess.get(api, params={"bvid": bvid}, timeout=10)
            j = r.json()
            if j.get("code") != 0:
                _sleep_jitter(0.6)
                continue
            d = j["data"]
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
            }
        except Exception:
            _sleep_jitter(0.6)
    return None


def fetch_vlist_by_mid(mid: int, n: int = 30, use_browser_fallback: bool = True, sleep_sec: float = 0.8) -> list[dict]:
    """
    KOL近期公开视频列表抓取优化版：
    1）先访问UP个人主页，建立更像真实浏览器的 Session/Cookie；
    2）优先使用 /x/space/wbi/arc/search；
    3）失败后尝试老接口；
    4）仍为空时，尝试从UP空间HTML提取BV；
    5）可选：如果环境支持 selenium+Chrome，则无头打开UP主页再用浏览器cookie重试。
    """
    api_wbi = "https://api.bilibili.com/x/space/wbi/arc/search"
    api_old = "https://api.bilibili.com/x/space/arc/search"
    space_url = f"https://space.bilibili.com/{mid}/video"

    sess = _make_bili_session(referer=space_url)
    try:
        sess.get(space_url, timeout=10)
    except Exception:
        pass

    ps = 50

    def _call(api_url: str, pn_: int, session: requests.Session) -> tuple[int, list]:
        params = {"mid": mid, "pn": pn_, "ps": ps, "order": "pubdate", "platform": "web", "web_location": "1550101"}
        if "wbi" in api_url:
            params = _wbi_sign(params)
        h = {"Referer": space_url}
        r = session.get(api_url, params=params, headers=h, timeout=12)
        j = r.json()
        code = j.get("code", -1)
        data = j.get("data") or {}
        vlist = ((data.get("list") or {}).get("vlist")) or []
        return code, vlist

    def _api_loop(session: requests.Session) -> list[dict]:
        rows = []
        pn = 1
        while len(rows) < n and pn <= 6:
            vlist = []
            try:
                code, vlist = _call(api_wbi, pn, session)
                if code != 0 or not vlist:
                    code2, vlist2 = _call(api_old, pn, session)
                    if code2 == 0 and vlist2:
                        vlist = vlist2
            except Exception:
                vlist = []

            if not vlist:
                break

            for v in vlist:
                row = _as_video_row(v)
                if row:
                    rows.append(row)
                if len(rows) >= n:
                    break
            pn += 1
            _sleep_jitter(sleep_sec)
        return _dedupe_rows(rows, n)

    # A. API + 主页cookie
    out = _api_loop(sess)
    if len(out) >= max(1, min(n, 3)):
        return out[:n]

    # B. 直接读取空间页HTML兜底
    try:
        html = sess.get(space_url, timeout=12).text
        html_rows = _extract_video_rows_from_html(html, n=n)
        out = _dedupe_rows(out + html_rows, n)
        if len(out) >= max(1, min(n, 3)):
            return out[:n]
    except Exception:
        pass

    # C. 可选无头浏览器打开UP主页，拿浏览器cookie后重试
    if use_browser_fallback:
        html, browser_cookies = _open_space_with_headless_browser(mid)
        if browser_cookies:
            for k, v in browser_cookies.items():
                try:
                    sess.cookies.set(k, v, domain=".bilibili.com")
                except Exception:
                    sess.cookies.set(k, v)
            more = _api_loop(sess)
            out = _dedupe_rows(out + more, n)
        if html:
            out = _dedupe_rows(out + _extract_video_rows_from_html(html, n=n), n)

    return out[:n]

# =========================
# KOL 标注
# =========================
def kol_flag(view_lift: float | None, er_lift: float | None, deep_lift: float | None) -> str:
    def _v(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return float(x)
        except Exception:
            return None
    v, e, d = _v(view_lift), _v(er_lift), _v(deep_lift)
    if (v is not None and v >= 0.30) or (e is not None and e >= 0.20) or (d is not None and d >= 0.10):
        return "⭐ 合作明显更好"
    if (v is not None and v <= -0.20) or (e is not None and e <= -0.15):
        return "⚠️ 合作偏弱"
    return ""

# =========================
# Sidebar - global settings
# =========================
st.sidebar.title("📊 B站运营Dashboard")
# ✅ 展示当前DB、备份位置 & 当前记录数（方便判断是不是“环境重置”）
try:
    _tmp = load_all_rows()
    st.sidebar.caption(f"DB: {DB_PATH}")
    st.sidebar.caption(f"Backup: {BACKUP_LATEST_CSV}")
    st.sidebar.caption(f"Rows: {0 if _tmp is None else len(_tmp)}")
except Exception:
    st.sidebar.caption(f"DB: {DB_PATH}")
    st.sidebar.caption(f"Backup: {BACKUP_LATEST_CSV}")

st.sidebar.markdown("#### 全局“发挥评价”口径（按KOL自身历史，不按时间）")
baseline_window_n = st.sidebar.slider("基准：取该KOL最近N条视频（按发布时间/抓取时间排序）", 10, 60, 20, step=5)
baseline_min_n = st.sidebar.slider("最低样本数（只与库内条数有关）", 1, 20, 6, step=1)

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
        st.success("已清空（备份文件未删除，如需可从备份恢复）。")
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
            detail = fetch_video_detail_by_bvid(bvid)
            if detail is None:
                fail += 1
                continue
            detail["project"] = project
            detail["url"] = it
            detail["data_type"] = "collab"
            detail["baseline_for"] = ""
            detail["fans_delta"] = 0
            detail["fetched_at"] = pd.Timestamp.now()
            rows.append(detail)
            ok += 1
            time.sleep(0.35)

        if rows:
            df_new = normalize_df(pd.DataFrame(rows))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（已保存+自动备份）")
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

                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（已保存+自动备份）")
                st.rerun()

# =========================
# Load data
# =========================
df_db = load_all_rows()
df_db = normalize_df(df_db) if not df_db.empty else df_db

st.title("B站日常运营数据 Dashboard")
if df_db.empty:
    st.info("数据库为空：请在左侧采集或导入。（若你认为数据不应为空：说明部署环境可能重置了磁盘；本应用会优先尝试从 backup/backup_latest.csv 自动恢复）")
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
# Add performance labels
# =========================
df_f = add_perf_cols(df_f, df_db, baseline_window_n, baseline_min_n)

# =========================
# KPI cards
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# =========================
# Cross project comparison + Quadrant
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

st.markdown("**项目四象限（X=互动率中位数，Y=深度信号中位数）**")
if len(proj_df) >= 2:
    x_med = float(proj_df["互动率中位数"].median())
    y_med = float(proj_df["深度信号中位数"].median())

    fig_q = px.scatter(
        proj_df,
        x="互动率中位数",
        y="深度信号中位数",
        size="总播放",
        text="project",
        hover_data=["视频数","UP数","总播放","Top1播放贡献","Top3播放贡献","互动率波动(IQR)"],
    )
    fig_q.add_vline(x=x_med, line_dash="dash")
    fig_q.add_hline(y=y_med, line_dash="dash")
    fig_q.update_traces(textposition="top center")
    fig_q.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig_q, use_container_width=True)

# =========================
# ✅ 跨项目解读（四象限下方：项目对比）
# =========================
st.subheader("跨项目解读（四象限下方：用于对比不同项目）")
if proj_df.empty:
    st.info("暂无项目数据可解读。")
else:
    p = proj_df.copy()
    p["er"] = p["互动率中位数"]
    p["deep"] = p["深度信号中位数"]
    p["iqr"] = p["互动率波动(IQR)"]
    p["top1"] = p["Top1播放贡献"]
    p["top3"] = p["Top3播放贡献"]

    strongest = p.sort_values(["er","deep"], ascending=False).head(1).iloc[0]
    steadiest = p.sort_values(["iqr","er"], ascending=[True, False]).head(1).iloc[0]
    risky = p.sort_values(["top1","iqr"], ascending=False).head(1).iloc[0]

    lines = []
    lines.append("1）整体结构：当前项目在四象限中呈现差异化分布，可采用不同内容打法与KPI重点。")
    lines.append(f"2）更强项目（互动&沉淀更靠前）：{strongest['project']}（互动率中位数 {strongest['er']*100:.2f}%，深度信号中位数 {strongest['deep']*100:.1f}%）。")
    lines.append(f"3）更稳项目（波动更小）：{steadiest['project']}（互动率波动IQR {steadiest['iqr']*100:.2f}pp）。")
    lines.append(f"4）结构风险提示：{risky['project']} Top1播放贡献 {risky['top1']*100:.1f}%（Top3 {risky['top3']*100:.1f}%），建议补齐腰部内容密度降低单点波动。")
    st.write("\n".join(lines))

# =========================
# 项目内视频表
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
# Top/Bottom 深挖
# =========================
st.subheader("Top / Bottom 深挖（含KOL自身基准判断）")
for proj in (sel_projects if sel_projects else projects):
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
# 箱线图
# =========================
st.subheader("互动率分布（项目/UP主快速定位异常）")
fig = px.box(df_f, x="project", y="engagement_rate", points="all", hover_data=["title","owner_name","view"])
st.plotly_chart(fig, use_container_width=True)

# =========================
# ✅ 周报结论（逐项目输出：只评判项目内）
# =========================
st.subheader("周报结论（逐项目输出：只评判项目内）")
projects_for_weekly = sel_projects if (sel_projects and len(sel_projects) > 0) else projects
if not projects_for_weekly:
    st.info("暂无项目可输出周报结论。")
else:
    blocks = []
    idx = 1
    for proj in projects_for_weekly:
        wk = df_f[df_f["project"] == proj].copy()
        if wk.empty:
            continue
        wk = wk.sort_values("view", ascending=False)

        total_view = int(wk["view"].sum())
        total_eng = int(wk["engagement"].sum())
        er_med = float(wk["engagement_rate"].median())
        deep_med = float(wk["deep_signal_ratio"].median())
        video_cnt = int(len(wk))
        up_cnt = int(wk["owner_name"].nunique())

        top = wk.iloc[0]
        bottom = wk.iloc[-1]
        top1_share = float(top["view"]) / total_view if total_view > 0 else 0.0
        top3_share = float(wk.head(3)["view"].sum()) / total_view if total_view > 0 else 0.0
        er_iqr = float(wk["engagement_rate"].quantile(0.75) - wk["engagement_rate"].quantile(0.25))

        lines = []
        lines.append(f"项目{idx}｜【{proj}】")
        lines.append(f"- 产出与规模：{video_cnt} 条内容 / {up_cnt} 位UP，累计播放 {total_view:,}，累计互动 {total_eng:,}。")
        lines.append(f"- 互动质量：互动率中位数 {er_med*100:.2f}%（波动IQR {er_iqr*100:.2f}pp），深度信号中位数 {deep_med*100:.1f}%。")
        lines.append(f"- 高表现样本：最高播放《{top['title']}》{int(top['view']):,} 播放，互动率 {top['engagement_rate']*100:.2f}%，具备可复用抓手。")
        lines.append(f"- 待优化样本：最低播放《{bottom['title']}》{int(bottom['view']):,} 播放，建议从封面/标题信息密度与评论区互动引导做轻量优化，抬升底盘。")
        lines.append(f"- 结构观察：Top1贡献 {top1_share*100:.1f}%（Top3 {top3_share*100:.1f}%），后续通过复用高表现模板+补齐腰部内容，降低单点波动。")

        blocks.append("\n".join(lines))
        idx += 1

    st.write("\n\n".join(blocks))

# =========================
# 保留：全局自动解读（原模块保留）
# =========================
st.subheader("全局自动解读（原模块保留）")
best = df_f.sort_values("view", ascending=False).iloc[0]
worst = df_f.sort_values("view", ascending=True).iloc[0]
insights = []
insights.append(
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放，{best['播放表现']}），互动率 {best['engagement_rate']*100:.2f}%（{best['互动率表现']}）。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放，{worst['播放表现']}），互动率 {worst['engagement_rate']*100:.2f}%（{worst['互动率表现']}）。建议检查封面/标题信息密度与投放时段，并在评论区做更强的互动引导。"
)
if df_f["deep_signal_ratio"].mean() < 0.35:
    insights.append("3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”，建议强化：价值点前置、结尾引导收藏/投币、增加系列化承诺。")
else:
    insights.append("3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性，可考虑围绕该方向做系列化与固定栏目节奏。")
st.write("\n".join(insights))

# =========================================================
# KOL module（按 owner_mid，对齐+补齐+标注+导出）
# =========================================================
st.divider()
st.subheader("KOL合作资料库（独立模块：标注合作是否优于平时｜按owner_mid对齐）")

with st.expander("KOL模块设置", expanded=False):
    collab_projects = st.multiselect("哪些项目算合作项目", projects, default=sel_projects if sel_projects else projects)
    fetch_n = st.slider("补齐基准：每个KOL抓取最近N条公开视频", 10, 80, 30, step=5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)
    use_browser_fallback = st.checkbox("启用类浏览器/无头浏览器兜底抓取UP主页（推荐）", value=True)
    show_kol_quality_hint = st.checkbox("显示数据质量提示（缺mid/异常mid）", value=False)

cA, cB, cC = st.columns([1, 1, 2])
with cA:
    btn_fill_all = st.button("🧲 一键补齐所有合作KOL基准（写入__BASELINE__）")
with cB:
    btn_build_kol = st.button("📚 生成KOL对比表（含标注）")
with cC:
    st.caption("本版关键：WBI签名 + 主页Cookie预热 + HTML提取BV + 可选无头浏览器兜底，降低‘主页有视频但接口为空’的问题。")

if collab_projects:
    collab_df = df_db[df_db["project"].isin(collab_projects)].copy()
    raw_mid = collab_df["owner_mid"].copy()
    raw_mid_str = raw_mid.astype(str).fillna("").str.strip()
    bad_mask = raw_mid_str.eq("") | raw_mid_str.str.contains(r"\D", regex=True) | raw_mid_str.str.len().gt(12)

    collab_df["owner_mid"] = collab_df["owner_mid"].apply(_norm_mid)
    valid_mid_df = collab_df[collab_df["owner_mid"].astype(str).str.len() > 0].copy()
    invalid_mid_cnt = int((collab_df["owner_mid"].astype(str).str.len() == 0).sum())

    st.caption(f"合作UP主数：{collab_df['owner_mid'].nunique()}（含缺/异常mid）｜可抓取mid的UP数：{valid_mid_df['owner_mid'].nunique()}｜合作视频数：{len(collab_df)}")

    if show_kol_quality_hint:
        bad_rows = collab_df[bad_mask.values].copy()
        if not bad_rows.empty:
            st.warning(f"发现 {len(bad_rows)} 条合作视频 owner_mid 缺失/异常（仅影响这些视频被纳入KOL对齐）。")
            st.dataframe(bad_rows[["project","bvid","title","owner_name","owner_mid"]], use_container_width=True, height=220)
        else:
            st.success("未发现合作视频的 owner_mid 异常。")

    name_map = (valid_mid_df.groupby("owner_mid")["owner_name"]
                .agg(lambda s: s.value_counts().index[0]).to_dict())

    if btn_fill_all:
        existed_baseline = set(df_db[df_db["project"] == BASELINE_PROJECT]["bvid"].astype(str).tolist())
        rows_to_write = {}
        stat = {
            "list_fail": 0, "list_empty": 0, "detail_ok": 0, "detail_fail": 0, "vlist_added": 0,
            "collab_refresh_total": 0, "collab_refresh_ok": 0, "collab_refresh_fail": 0
        }

        # ========= A) 补齐 baseline =========
        for mid in sorted(valid_mid_df["owner_mid"].unique().tolist()):
            disp = name_map.get(mid, "")
            try:
                vlist = fetch_vlist_by_mid(int(mid), n=int(fetch_n), use_browser_fallback=bool(use_browser_fallback), sleep_sec=float(sleep_sec))
            except Exception:
                stat["list_fail"] += 1
                continue

            if not vlist:
                stat["list_empty"] += 1
                continue

            for v in vlist:
                bvid = v["bvid"]
                if bvid in existed_baseline:
                    continue

                base_row = {
                    "project": BASELINE_PROJECT,
                    "bvid": bvid,
                    "url": f"https://www.bilibili.com/video/{bvid}",
                    "title": v.get("title",""),
                    "pubdate": v.get("pubdate", pd.NaT),
                    "owner_mid": mid,
                    "owner_name": disp,
                    "view": _safe_int(v.get("view",0)),
                    "reply": _safe_int(v.get("reply",0)),
                    "like": 0, "coin": 0, "favorite": 0, "danmaku": 0, "share": 0,
                    "fans_delta": 0,
                    "baseline_for": disp,
                    "data_type": "baseline",
                    "fetched_at": pd.Timestamp.now(),
                }
                rows_to_write[(BASELINE_PROJECT, bvid)] = base_row
                existed_baseline.add(bvid)
                stat["vlist_added"] += 1

                detail = fetch_video_detail_by_bvid(bvid)
                if detail is not None:
                    detail["project"] = BASELINE_PROJECT
                    detail["url"] = f"https://www.bilibili.com/video/{bvid}"
                    detail["baseline_for"] = disp
                    detail["data_type"] = "baseline"
                    detail["fans_delta"] = 0
                    detail["fetched_at"] = pd.Timestamp.now()
                    rows_to_write[(BASELINE_PROJECT, bvid)] = detail
                    stat["detail_ok"] += 1
                else:
                    stat["detail_fail"] += 1

                time.sleep(float(sleep_sec))

        # ========= B) 刷新已入库合作BV =========
        valid_mids = set(valid_mid_df["owner_mid"].astype(str).tolist())
        collab_in_db = df_db[(df_db["project"].isin(collab_projects)) & (df_db["project"] != BASELINE_PROJECT)].copy()
        collab_in_db["owner_mid"] = collab_in_db["owner_mid"].apply(_norm_mid)
        collab_in_db = collab_in_db[collab_in_db["owner_mid"].isin(valid_mids)]
        collab_pairs = (collab_in_db[["project","bvid","url"]]
                        .dropna(subset=["project","bvid"])
                        .drop_duplicates()
                        .values
                        .tolist())
        stat["collab_refresh_total"] = len(collab_pairs)

        for proj, bvid, url in collab_pairs:
            bvid = str(bvid)
            if not bvid.startswith("BV"):
                continue

            detail = fetch_video_detail_by_bvid(bvid)
            if detail is None:
                stat["collab_refresh_fail"] += 1
                time.sleep(float(sleep_sec))
                continue

            detail["project"] = str(proj)
            detail["url"] = _safe_str(url) if _safe_str(url) else f"https://www.bilibili.com/video/{bvid}"
            detail["data_type"] = "collab"
            detail["baseline_for"] = ""
            detail["fans_delta"] = 0
            detail["fetched_at"] = pd.Timestamp.now()
            rows_to_write[(detail["project"], bvid)] = detail
            stat["collab_refresh_ok"] += 1
            time.sleep(float(sleep_sec))

        # ========= C) 写入数据库 =========
        if rows_to_write:
            df_new = normalize_df(pd.DataFrame(list(rows_to_write.values())))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

            st.success(
                f"补齐完成：新增 {stat['vlist_added']} 条基准；"
                f"列表失败 {stat['list_fail']}，列表空 {stat['list_empty']}；"
                f"详情补全成功 {stat['detail_ok']}，失败 {stat['detail_fail']}；"
                f"合作BV刷新：{stat['collab_refresh_ok']}/{stat['collab_refresh_total']} 成功（失败 {stat['collab_refresh_fail']}）。"
                f"（数据已自动备份到 backup/backup_latest.csv）"
            )
            st.rerun()
        else:
            st.warning("本次未写入任何数据：可能接口波动导致vlist为空，且合作BV刷新也未成功。")

    st.markdown("**KOL基准诊断（按owner_mid统计库内数量）**")
    diag = []
    for mid in sorted(valid_mid_df["owner_mid"].unique().tolist()):
        owner_all = df_db[df_db["owner_mid"].apply(_norm_mid) == mid].copy()
        owner_all = _sort_owner_hist(owner_all)

        base_pool = owner_all[~owner_all["project"].isin(set(collab_projects))].copy()
        base_pool = pd.concat([base_pool, owner_all[owner_all["project"] == BASELINE_PROJECT]], ignore_index=True)
        base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")
        base_pool = _sort_owner_hist(base_pool)

        avail = int(min(len(base_pool), baseline_window_n))
        diag.append({
            "owner_mid": mid,
            "KOL/UP主": name_map.get(mid, owner_all["owner_name"].dropna().iloc[0] if not owner_all.empty else ""),
            "库内视频总数": int(len(owner_all)),
            "可用基准数(平时池)": int(len(base_pool)),
            f"取最近{baseline_window_n}可用": avail,
            "状态": "OK" if avail >= baseline_min_n else f"基准不足(<{baseline_min_n})",
        })
    st.dataframe(pd.DataFrame(diag).sort_values(["状态","可用基准数(平时池)"], ascending=[True, False]),
                 use_container_width=True, height=360)

    if btn_build_kol:
        df_all_m = compute_metrics(df_db.copy())
        df_all_m["owner_mid"] = df_all_m["owner_mid"].apply(_norm_mid)
        rows = []

        collab_mid_df = df_all_m[df_all_m["project"].isin(collab_projects)].copy()
        collab_mid_df = collab_mid_df[collab_mid_df["owner_mid"].astype(str).str.len() > 0]

        for mid, g_collab in collab_mid_df.groupby("owner_mid"):
            up_name = (g_collab["owner_name"].value_counts().index[0]
                       if not g_collab["owner_name"].dropna().empty else name_map.get(mid, ""))

            owner_all = df_all_m[df_all_m["owner_mid"] == mid].copy()
            owner_all = _sort_owner_hist(owner_all)

            base_pool = owner_all[~owner_all["project"].isin(set(collab_projects))].copy()
            base_pool = pd.concat([base_pool, owner_all[owner_all["project"] == BASELINE_PROJECT]], ignore_index=True)
            base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")
            base_pool = _sort_owner_hist(base_pool).head(baseline_window_n)

            if len(base_pool) < baseline_min_n:
                continue

            base_view = float(base_pool["view"].median())
            base_er = float(base_pool["engagement_rate"].median())
            base_deep = float(base_pool["deep_signal_ratio"].median())

            collab_view = float(g_collab["view"].median())
            collab_er = float(g_collab["engagement_rate"].median())
            collab_deep = float(g_collab["deep_signal_ratio"].median())

            view_lift = (collab_view / base_view - 1.0) if base_view > 0 else np.nan
            er_lift = (collab_er / base_er - 1.0) if base_er > 0 else np.nan
            deep_lift = (collab_deep / base_deep - 1.0) if base_deep > 0 else np.nan

            mark = kol_flag(view_lift, er_lift, deep_lift)

            tags = []
            if not np.isnan(view_lift) and view_lift >= 0.30: tags.append("热度拉升")
            if not np.isnan(er_lift) and er_lift >= 0.20: tags.append("互动增强")
            if not np.isnan(deep_lift) and deep_lift >= 0.10: tags.append("沉淀提升")
            if not tags: tags.append("常规")

            persona = f"{'热度拉升' if '热度拉升' in tags else '热度稳定'} + {'互动增强' if '互动增强' in tags else '互动常规'} + {'沉淀提升' if '沉淀提升' in tags else '沉淀一般'}"

            rows.append({
                "owner_mid": mid,
                "KOL/UP主": up_name,
                "标注": mark,
                "合作视频数": int(len(g_collab)),
                "基准样本数": int(len(base_pool)),
                "标签": "、".join(tags),
                "KOL画像一句话": persona,
                "合作播放中位数": int(collab_view),
                "基准播放中位数": int(base_view),
                "播放提升": "-" if np.isnan(view_lift) else f"{view_lift*100:.1f}%",
                "合作互动率中位数": f"{collab_er*100:.2f}%",
                "基准互动率中位数": f"{base_er*100:.2f}%",
                "互动率提升": "-" if np.isnan(er_lift) else f"{er_lift*100:.1f}%",
                "合作深度信号中位数": f"{collab_deep*100:.1f}%",
                "基准深度信号中位数": f"{base_deep*100:.1f}%",
                "深度信号提升": "-" if np.isnan(deep_lift) else f"{deep_lift*100:.1f}%"
            })

        if not rows:
            st.warning("没有生成KOL结果：请先补齐基准，或降低最低样本数。")
        else:
            lib = pd.DataFrame(rows)
            st.dataframe(lib, use_container_width=True, height=520)
            st.download_button(
                "⬇️ 下载KOL对比表（CSV）",
                data=lib.to_csv(index=False).encode("utf-8-sig"),
                file_name="kol_compare.csv",
                mime="text/csv"
            )
