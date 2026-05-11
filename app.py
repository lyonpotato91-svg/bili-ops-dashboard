
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
        "昵称": "owner_name",
        "账号昵称": "owner_name",
        "UP昵称": "owner_name",
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
        "视频BV": "bvid",
        "视频BV号": "bvid",
        "视频BV链接": "url",
        "BV链接": "url",
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
    if not img_key or not sub_key:
        raise RuntimeError("未获取到 WBI img_key/sub_key")
    return img_key, sub_key

def _wbi_sign(params: dict) -> dict:
    img_key, sub_key = _get_wbi_keys()
    mixin_key = _get_mixin_key(img_key, sub_key)

    params = {k: v for k, v in params.items() if v is not None}
    params["wts"] = int(time.time())

    def _filter(v):
        return re.sub(r"[!'()*]", "", str(v))

    sorted_items = sorted((k, _filter(v)) for k, v in params.items())
    # WBI 更接近 encodeURIComponent：空格用 %20，而不是 application/x-www-form-urlencoded 的 +
    query = urllib.parse.urlencode(sorted_items, quote_via=urllib.parse.quote)
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


def _apply_cookie_to_session(sess: requests.Session, cookie: str = "") -> requests.Session:
    cookie = (cookie or "").strip()
    if cookie:
        # 只在用户主动提供时使用；避免在代码里硬编码敏感 cookie
        sess.headers.update({"Cookie": cookie})
    return sess


def _normalize_proxy(proxy: str = "") -> str:
    """把用户输入的代理地址规范化；留空则不使用代理。"""
    proxy = (proxy or "").strip()
    if not proxy:
        return ""
    if proxy.startswith(("http://", "https://", "socks5://", "socks5h://")):
        return proxy
    return "http://" + proxy


def _apply_proxy_to_session(sess: requests.Session, proxy: str = "") -> requests.Session:
    proxy = _normalize_proxy(proxy)
    if proxy:
        sess.proxies.update({"http": proxy, "https": proxy})
    return sess


def _make_bili_session(referer: str = "https://www.bilibili.com/", cookie: str = "", proxy: str = "") -> requests.Session:
    """
    为每次抓取建立带常见浏览器头、可选 Cookie、可选代理的 Session。
    Cookie/代理只在用户主动填写时生效，避免把敏感信息写入代码。
    """
    sess = requests.Session()
    h = HEADERS.copy()
    h.update({
        "Referer": referer,
        "Origin": "https://www.bilibili.com",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    })
    sess.headers.update(h)
    _apply_cookie_to_session(sess, cookie)
    _apply_proxy_to_session(sess, proxy)
    try:
        sess.get("https://www.bilibili.com/", timeout=8)
    except Exception:
        # 预热失败不应让页面崩溃；后续接口诊断会记录真实失败原因
        pass
    return sess


def _bili_num_to_int(x, default=0) -> int:
    """兼容 1.2万、3亿、1,234、-- 等 B站常见数字格式。"""
    try:
        if x is None or pd.isna(x):
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return int(x)
        s = str(x).strip().replace(",", "")
        if not s or s in ["--", "-", "nan", "None"]:
            return default
        m = re.search(r"([\d.]+)\s*([万亿wWkK]?)", s)
        if not m:
            return default
        num = float(m.group(1))
        unit = m.group(2)
        if unit in ["万", "w", "W"]:
            num *= 10000
        elif unit == "亿":
            num *= 100000000
        elif unit in ["k", "K"]:
            num *= 1000
        return int(num)
    except Exception:
        return default


def _clean_bili_title(x) -> str:
    s = _safe_str(x)
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&quot;", '"').replace("&amp;", "&").strip()


def _as_video_row(v: dict) -> dict | None:
    """兼容 B站 Web / APP / 页面 JSON 里常见的视频字段。"""
    if not isinstance(v, dict):
        return None

    bvid = v.get("bvid") or v.get("bvidStr") or v.get("BV")
    if not bvid:
        uri = _safe_str(v.get("uri") or v.get("arcurl") or v.get("url") or v.get("jump_url") or "")
        bvid = parse_bvid(uri)
    if not bvid:
        return None

    created = (
        v.get("created")
        or v.get("pubdate")
        or v.get("ctime")
        or v.get("ptime")
        or v.get("publish_time")
        or v.get("pub_time")
    )

    pubdate = pd.NaT
    if created is not None and str(created).strip() != "":
        try:
            # 秒级时间戳才走 unit=s；其它字符串交给 pandas 自己识别
            created_i = _safe_int(created, default=-1)
            if created_i >= 1000000000:
                pubdate = pd.to_datetime(created_i, unit="s", errors="coerce")
            else:
                pubdate = pd.to_datetime(created, errors="coerce")
        except Exception:
            pubdate = pd.NaT

    stat = v.get("stat") or v.get("stats") or {}
    return {
        "bvid": bvid,
        "title": _clean_bili_title(v.get("title", "")),
        "pubdate": pubdate,
        "view": _bili_num_to_int(v.get("play", stat.get("view", v.get("view", v.get("view_content", 0))))),
        "like": _bili_num_to_int(v.get("like", stat.get("like", 0))),
        "coin": _bili_num_to_int(v.get("coin", stat.get("coin", 0))),
        "favorite": _bili_num_to_int(v.get("favorite", stat.get("favorite", 0))),
        "reply": _bili_num_to_int(v.get("comment", stat.get("reply", v.get("reply", 0)))),
        "danmaku": _bili_num_to_int(v.get("danmaku", stat.get("danmaku", 0))),
        "share": _bili_num_to_int(v.get("share", stat.get("share", 0))),
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


def _log_kol_debug(debug: list | None, source: str, ok: bool, msg: str = "", count: int = 0, code=None):
    if debug is None:
        return
    debug.append({
        "source": source,
        "ok": bool(ok),
        "count": int(count or 0),
        "code": "" if code is None else code,
        "message": _safe_str(msg)[:220],
        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
    })


def _extract_video_rows_from_html(html: str, n: int = 30, cookie: str = "", proxy: str = "") -> list[dict]:
    """从UP空间页源码里兜底提取 BV 号，再用详情接口补数据。"""
    if not html:
        return []

    bvids = []
    # 先尝试解析 __INITIAL_STATE__ 中的 BV；失败则用正则扫全页
    for bv in re.findall(r"BV[0-9A-Za-z]{10}", html):
        if bv not in bvids:
            bvids.append(bv)
        if len(bvids) >= n:
            break

    rows = []
    sess = _make_bili_session(cookie=cookie, proxy=proxy)
    for bvid in bvids[:n]:
        detail = fetch_video_detail_by_bvid(bvid, sess=sess)
        if detail is not None:
            rows.append({
                "bvid": bvid,
                "title": detail.get("title", ""),
                "pubdate": detail.get("pubdate", pd.NaT),
                "view": _safe_int(detail.get("view", 0)),
                "reply": _safe_int(detail.get("reply", 0)),
                "like": _safe_int(detail.get("like", 0)),
                "coin": _safe_int(detail.get("coin", 0)),
                "favorite": _safe_int(detail.get("favorite", 0)),
                "danmaku": _safe_int(detail.get("danmaku", 0)),
                "share": _safe_int(detail.get("share", 0)),
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


def fetch_video_detail_by_bvid(bvid: str, sess: requests.Session | None = None, cookie: str = "", proxy: str = "") -> dict | None:
    api = "https://api.bilibili.com/x/web-interface/view"
    close_session = False
    if sess is None:
        sess = _make_bili_session(referer=f"https://www.bilibili.com/video/{bvid}", cookie=cookie, proxy=proxy)
        close_session = True

    try:
        for _ in range(3):
            try:
                r = sess.get(api, params={"bvid": bvid}, headers={"Referer": f"https://www.bilibili.com/video/{bvid}"}, timeout=10)
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
    finally:
        if close_session:
            try:
                sess.close()
            except Exception:
                pass
    return None


def _fetch_vlist_by_mid_web_wbi(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """Web 端 UP 投稿列表：主路径。"""
    api = "https://api.bilibili.com/x/space/wbi/arc/search"
    space_url = f"https://space.bilibili.com/{mid}/video"
    rows = []
    ps = min(50, max(10, int(n)))

    for pn in range(1, 8):
        if len(rows) >= n:
            break
        params = {
            "mid": mid,
            "pn": pn,
            "ps": ps,
            "tid": 0,
            "keyword": "",
            "order": "pubdate",
            "order_avoided": "true",
            "platform": "web",
            "web_location": "1550101",
        }
        try:
            signed = _wbi_sign(params)
            r = sess.get(api, params=signed, headers={"Referer": space_url}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            vlist = ((data.get("list") or {}).get("vlist")) or []
            if code == 0 and vlist:
                for v in vlist:
                    row = _as_video_row(v)
                    if row:
                        rows.append(row)
                _log_kol_debug(debug, "web_wbi_arc_search", True, f"pn={pn}", len(vlist), code)
            else:
                msg = j.get("message") or data.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "web_wbi_arc_search", False, f"pn={pn}: {msg}", 0, code)
                if pn == 1:
                    break
        except Exception as e:
            _log_kol_debug(debug, "web_wbi_arc_search", False, f"pn={pn}: {type(e).__name__}: {e}", 0, "EXC")
            if pn == 1:
                break
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_web_old(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """老 Web 投稿接口：保留兜底。"""
    api = "https://api.bilibili.com/x/space/arc/search"
    space_url = f"https://space.bilibili.com/{mid}/video"
    rows = []
    ps = min(50, max(10, int(n)))

    for pn in range(1, 6):
        if len(rows) >= n:
            break
        params = {
            "mid": mid,
            "pn": pn,
            "ps": ps,
            "tid": 0,
            "keyword": "",
            "order": "pubdate",
        }
        try:
            r = sess.get(api, params=params, headers={"Referer": space_url}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            vlist = ((data.get("list") or {}).get("vlist")) or []
            if code == 0 and vlist:
                for v in vlist:
                    row = _as_video_row(v)
                    if row:
                        rows.append(row)
                _log_kol_debug(debug, "web_old_arc_search", True, f"pn={pn}", len(vlist), code)
            else:
                msg = j.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "web_old_arc_search", False, f"pn={pn}: {msg}", 0, code)
                if pn == 1:
                    break
        except Exception as e:
            _log_kol_debug(debug, "web_old_arc_search", False, f"pn={pn}: {type(e).__name__}: {e}", 0, "EXC")
            if pn == 1:
                break
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_app_cursor(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    APP 端 cursor 投稿接口兜底：无需 WBI，字段里直接给 bvid/play/danmaku/ctime。
    对 web arc/search 返回空或被风控时更稳。
    """
    api_candidates = [
        "https://app.biliapi.com/x/v2/space/archive/cursor",
        "https://app.bilibili.com/x/v2/space/archive/cursor",
    ]
    rows = []
    ps = min(50, max(10, int(n)))
    last_aid = None

    for api in api_candidates:
        rows = []
        last_aid = None
        for page_i in range(1, 8):
            if len(rows) >= n:
                break
            params = {
                "vmid": mid,
                "ps": ps,
                "order": "pubdate",
                "platform": "web",
                "mobi_app": "web",
                "fnver": 0,
                "fnval": 4048,
                "fourk": 1,
                "ts": int(time.time()),
            }
            if last_aid:
                params["aid"] = last_aid

            try:
                r = sess.get(api, params=params, headers={"Referer": f"https://space.bilibili.com/{mid}/video"}, timeout=12)
                j = r.json()
                code = j.get("code", -1)
                data = j.get("data") or {}
                item = data.get("item") or data.get("items") or []
                if code == 0 and item:
                    for v in item:
                        row = _as_video_row(v)
                        if row:
                            rows.append(row)
                    last = item[-1] if item else {}
                    last_aid = last.get("param") or last.get("aid") or last.get("id")
                    _log_kol_debug(debug, "app_archive_cursor", True, f"{api.split('/')[2]} page={page_i}", len(item), code)
                    if not data.get("has_next", False) or not last_aid:
                        break
                else:
                    msg = j.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                    _log_kol_debug(debug, "app_archive_cursor", False, f"{api.split('/')[2]} page={page_i}: {msg}", 0, code)
                    break
            except Exception as e:
                _log_kol_debug(debug, "app_archive_cursor", False, f"{api.split('/')[2]} page={page_i}: {type(e).__name__}: {e}", 0, "EXC")
                break

            _sleep_jitter(sleep_sec)

        rows = _dedupe_rows(rows, n)
        if rows:
            return rows

    return []


def _extract_bvids_recursive(obj, limit: int = 120) -> list[str]:
    """从任意 JSON/HTML 片段递归提取 BV，供动态/合集/搜索兜底使用。"""
    out, seen = [], set()

    def add_from_text(text: str):
        for bv in re.findall(r"BV[0-9A-Za-z]{10}", _safe_str(text)):
            if bv not in seen:
                seen.add(bv)
                out.append(bv)
                if len(out) >= limit:
                    return

    def walk(x):
        if len(out) >= limit:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in {"bvid", "bvidstr"}:
                    add_from_text(v)
                else:
                    walk(v)
                if len(out) >= limit:
                    return
        elif isinstance(x, (list, tuple)):
            for item in x:
                walk(item)
                if len(out) >= limit:
                    return
        elif isinstance(x, str):
            add_from_text(x)

    walk(obj)
    return out[:limit]


def _detail_rows_from_bvids(
    bvids: list[str],
    owner_mid: int | str,
    n: int,
    sess: requests.Session,
    sleep_sec: float,
    debug: list | None = None,
    source: str = "detail_filter",
) -> list[dict]:
    """用视频详情接口反查 owner_mid，只保留目标 UP 自己的视频，避免搜索/动态串号。"""
    target_mid = _norm_mid(owner_mid)
    rows = []
    seen = set()
    for bvid in bvids:
        if len(rows) >= n:
            break
        bvid = parse_bvid(_safe_str(bvid)) or _safe_str(bvid)
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        detail = fetch_video_detail_by_bvid(bvid, sess=sess)
        if detail is None:
            _log_kol_debug(debug, source, False, f"detail_fail {bvid}", 0, "DETAIL")
            _sleep_jitter(min(float(sleep_sec), 0.8))
            continue
        got_mid = _norm_mid(detail.get("owner_mid", ""))
        if target_mid and got_mid != target_mid:
            _log_kol_debug(debug, source, False, f"skip_other_owner {bvid} owner_mid={got_mid}", 0, "FILTER")
            _sleep_jitter(min(float(sleep_sec), 0.8))
            continue
        rows.append({
            "bvid": detail.get("bvid", bvid),
            "title": detail.get("title", ""),
            "pubdate": detail.get("pubdate", pd.NaT),
            "view": _safe_int(detail.get("view", 0)),
            "reply": _safe_int(detail.get("reply", 0)),
            "like": _safe_int(detail.get("like", 0)),
            "coin": _safe_int(detail.get("coin", 0)),
            "favorite": _safe_int(detail.get("favorite", 0)),
            "danmaku": _safe_int(detail.get("danmaku", 0)),
            "share": _safe_int(detail.get("share", 0)),
        })
        _sleep_jitter(min(float(sleep_sec), 0.8))
    _log_kol_debug(debug, source, bool(rows), f"verified={len(rows)}", len(rows), "OK" if rows else "EMPTY")
    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_collection(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """合集/系列兜底。部分 UP 的普通投稿接口为空，但合集页仍能暴露 archives。"""
    rows = []
    space_url = f"https://space.bilibili.com/{mid}/video"
    list_api = "https://api.bilibili.com/x/polymer/web-space/seasons_series_list"
    archive_api = "https://api.bilibili.com/x/polymer/web-space/seasons_archives_list"
    try:
        params = _wbi_sign({"mid": mid, "page_num": 1, "page_size": 20, "web_location": "333.1387"})
        r = sess.get(list_api, params=params, headers={"Referer": space_url}, timeout=12)
        j = r.json()
        code = j.get("code", -1)
        data = j.get("data") or {}
        bvids = _extract_bvids_recursive(data, limit=n)
        if bvids:
            rows.extend(_detail_rows_from_bvids(bvids, mid, n - len(rows), sess, sleep_sec, debug, source="collection_list_detail_filter"))
        # 如果 list 内没有直接给 BV，再尝试 season/series archive 子接口
        ids = []
        for key in ["seasons_list", "series_list", "items_lists", "items"]:
            part = data.get(key) if isinstance(data, dict) else None
            if isinstance(part, dict):
                for sub in part.values():
                    if isinstance(sub, list):
                        for it in sub:
                            if isinstance(it, dict):
                                sid = it.get("season_id") or it.get("series_id") or it.get("id") or it.get("meta_id")
                                if sid and sid not in ids:
                                    ids.append(sid)
            elif isinstance(part, list):
                for it in part:
                    if isinstance(it, dict):
                        sid = it.get("season_id") or it.get("series_id") or it.get("id") or it.get("meta_id")
                        if sid and sid not in ids:
                            ids.append(sid)
        for sid in ids[:8]:
            if len(rows) >= n:
                break
            for pn in range(1, 4):
                if len(rows) >= n:
                    break
                try:
                    p = _wbi_sign({"mid": mid, "season_id": sid, "series_id": sid, "page_num": pn, "page_size": 30, "sort_reverse": "false", "web_location": "333.1387"})
                    rr = sess.get(archive_api, params=p, headers={"Referer": space_url}, timeout=12)
                    jj = rr.json()
                    bvs = _extract_bvids_recursive(jj.get("data") or jj, limit=n)
                    if not bvs:
                        break
                    rows.extend(_detail_rows_from_bvids(bvs, mid, n - len(rows), sess, sleep_sec, debug, source="collection_archive_detail_filter"))
                    _sleep_jitter(sleep_sec)
                except Exception as e:
                    _log_kol_debug(debug, "collection_archive", False, f"sid={sid} pn={pn}: {type(e).__name__}: {e}", 0, "EXC")
                    break
        _log_kol_debug(debug, "collection_series", bool(rows), f"code={code}; rows={len(rows)}", len(rows), code)
    except Exception as e:
        _log_kol_debug(debug, "collection_series", False, f"{type(e).__name__}: {e}", 0, "EXC")
    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_dynamic(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """空间动态兜底：动态里常含投稿卡片；最终仍用详情接口校验 owner_mid。"""
    api = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
    space_url = f"https://space.bilibili.com/{mid}/dynamic"
    rows = []
    offset = ""
    for page_i in range(1, 6):
        if len(rows) >= n:
            break
        params = {"host_mid": mid, "timezone_offset": -480, "features": "itemOpusStyle,listOnlyfans,opusBigCover,onlyfansVote", "web_location": "333.999"}
        if offset:
            params["offset"] = offset
        try:
            signed = _wbi_sign(params)
            r = sess.get(api, params=signed, headers={"Referer": space_url}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            items = data.get("items") or []
            bvids = _extract_bvids_recursive(items, limit=n * 2)
            if bvids:
                rows.extend(_detail_rows_from_bvids(bvids, mid, n - len(rows), sess, sleep_sec, debug, source="dynamic_detail_filter"))
            _log_kol_debug(debug, "dynamic_space", bool(bvids), f"page={page_i}; bvids={len(bvids)}", len(bvids), code)
            offset = data.get("offset") or data.get("history_offset") or ""
            if not offset or not data.get("has_more", False):
                break
        except Exception as e:
            _log_kol_debug(debug, "dynamic_space", False, f"page={page_i}: {type(e).__name__}: {e}", 0, "EXC")
            break
        _sleep_jitter(sleep_sec)
    return _dedupe_rows(rows, n)


def _fetch_vlist_by_name_search(owner_name: str, mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """昵称搜索兜底：搜索召回后用详情接口按 owner_mid 严格过滤。"""
    owner_name = _safe_str(owner_name).strip()
    if not owner_name:
        _log_kol_debug(debug, "name_search", False, "empty owner_name", 0, "SKIP")
        return []
    api = "https://api.bilibili.com/x/web-interface/wbi/search/type"
    rows, candidates = [], []
    for page_i in range(1, 6):
        if len(rows) >= n:
            break
        params = {
            "search_type": "video",
            "keyword": owner_name,
            "page": page_i,
            "page_size": 30,
            "order": "pubdate",
            "duration": 0,
            "tids": 0,
            "web_location": "1430654",
        }
        try:
            signed = _wbi_sign(params)
            r = sess.get(api, params=signed, headers={"Referer": "https://search.bilibili.com/"}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            result = data.get("result") or []
            for item in result:
                # result 里有时带 mid/author；能快速过滤就先过滤，不能则交给详情接口
                item_mid = _norm_mid(item.get("mid", "")) if isinstance(item, dict) else ""
                if item_mid and item_mid != _norm_mid(mid):
                    continue
                row = _as_video_row(item)
                if row and row.get("bvid"):
                    candidates.append(row["bvid"])
            _log_kol_debug(debug, "name_search", bool(result), f"page={page_i}; result={len(result)}; candidates={len(candidates)}", len(result), code)
            if not result:
                break
            rows = _dedupe_rows(rows + _detail_rows_from_bvids(candidates, mid, n - len(rows), sess, sleep_sec, debug, source="search_detail_filter"), n)
        except Exception as e:
            _log_kol_debug(debug, "name_search", False, f"page={page_i}: {type(e).__name__}: {e}", 0, "EXC")
            break
        _sleep_jitter(sleep_sec)
    return _dedupe_rows(rows, n)


def fetch_vlist_by_mid(
    mid: int,
    n: int = 30,
    use_browser_fallback: bool = False,
    sleep_sec: float = 0.8,
    cookie: str = "",
    proxy: str = "",
    owner_name: str = "",
    debug: list | None = None,
) -> list[dict]:
    """
    KOL近期公开视频列表抓取增强版：完整聚合多路径，直到补满 n 条才停止。\n
    路径：Web WBI投稿 → 老投稿 → APP cursor → 合集/系列 → 动态流 → 昵称搜索 → HTML BV → Selenium。
    所有非投稿列表来源都会通过视频详情反查 owner_mid，避免串号。
    """
    space_url = f"https://space.bilibili.com/{mid}/video"
    sess = _make_bili_session(referer=space_url, cookie=cookie, proxy=proxy)

    try:
        r = sess.get(space_url, timeout=10)
        _log_kol_debug(debug, "space_warmup", r.status_code == 200, f"HTTP {r.status_code}", 0, r.status_code)
    except Exception as e:
        _log_kol_debug(debug, "space_warmup", False, f"{type(e).__name__}: {e}", 0, "EXC")

    out = []
    fetchers = [
        ("web_wbi", lambda: _fetch_vlist_by_mid_web_wbi(mid, n - len(out), sess, sleep_sec, debug)),
        ("web_old", lambda: _fetch_vlist_by_mid_web_old(mid, n - len(out), sess, sleep_sec, debug)),
        ("app_cursor", lambda: _fetch_vlist_by_mid_app_cursor(mid, n - len(out), sess, sleep_sec, debug)),
        ("collection", lambda: _fetch_vlist_by_mid_collection(mid, n - len(out), sess, sleep_sec, debug)),
        ("dynamic", lambda: _fetch_vlist_by_mid_dynamic(mid, n - len(out), sess, sleep_sec, debug)),
        ("name_search", lambda: _fetch_vlist_by_name_search(owner_name, mid, n - len(out), sess, sleep_sec, debug)),
    ]
    for source_name, fn in fetchers:
        if len(out) >= n:
            break
        try:
            more = fn()
            out = _dedupe_rows(out + more, n)
            _log_kol_debug(debug, f"aggregate_{source_name}", bool(more), f"added={len(more)} total={len(out)}/{n}", len(more), "OK" if more else "EMPTY")
        except Exception as e:
            _log_kol_debug(debug, f"aggregate_{source_name}", False, f"{type(e).__name__}: {e}", 0, "EXC")
        _sleep_jitter(min(float(sleep_sec), 0.8))

    # HTML 兜底：只提 BV，再用详情校验 owner_mid
    if len(out) < n:
        try:
            html = sess.get(space_url, timeout=12).text
            html_bvids = _extract_bvids_recursive(html, limit=n * 2)
            html_rows = _detail_rows_from_bvids(html_bvids, mid, n - len(out), sess, sleep_sec, debug, source="space_html_detail_filter")
            out = _dedupe_rows(out + html_rows, n)
            _log_kol_debug(debug, "space_html_bv_extract", bool(html_rows), f"html_bv={len(html_bvids)}; added={len(html_rows)}", len(html_rows), "OK" if html_rows else "EMPTY")
        except Exception as e:
            _log_kol_debug(debug, "space_html_bv_extract", False, f"{type(e).__name__}: {e}", 0, "EXC")

    if use_browser_fallback and len(out) < n:
        html, browser_cookies = _open_space_with_headless_browser(mid)
        if browser_cookies:
            for k, v in browser_cookies.items():
                try:
                    sess.cookies.set(k, v, domain=".bilibili.com")
                except Exception:
                    sess.cookies.set(k, v)
            more = _fetch_vlist_by_mid_web_wbi(mid, n - len(out), sess, sleep_sec, debug)
            out = _dedupe_rows(out + more, n)
        if html and len(out) < n:
            html_bvids = _extract_bvids_recursive(html, limit=n * 2)
            html_rows = _detail_rows_from_bvids(html_bvids, mid, n - len(out), sess, sleep_sec, debug, source="selenium_html_detail_filter")
            out = _dedupe_rows(out + html_rows, n)
            _log_kol_debug(debug, "selenium_html_bv_extract", bool(html_rows), f"html_bv={len(html_bvids)}; added={len(html_rows)}", len(html_rows), "OK" if html_rows else "EMPTY")

    return out[:n]

def _detail_row_for_project(detail: dict, project: str, url: str = "", data_type: str = "collab", baseline_for: str = "") -> dict:
    """把详情接口结果统一转成数据库行。"""
    bvid = detail.get("bvid", "")
    return {
        "project": project,
        "bvid": bvid,
        "url": _safe_str(url) if _safe_str(url) else f"https://www.bilibili.com/video/{bvid}",
        "title": detail.get("title", ""),
        "pubdate": detail.get("pubdate", pd.NaT),
        "owner_mid": _norm_mid(detail.get("owner_mid", "")),
        "owner_name": detail.get("owner_name", ""),
        "view": _safe_int(detail.get("view", 0)),
        "like": _safe_int(detail.get("like", 0)),
        "coin": _safe_int(detail.get("coin", 0)),
        "favorite": _safe_int(detail.get("favorite", 0)),
        "reply": _safe_int(detail.get("reply", 0)),
        "danmaku": _safe_int(detail.get("danmaku", 0)),
        "share": _safe_int(detail.get("share", 0)),
        "fans_delta": 0,
        "baseline_for": baseline_for,
        "data_type": data_type,
        "fetched_at": pd.Timestamp.now(),
    }

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
                raw_import_count = len(df_csv)
                df_csv = normalize_df(df_csv)
                dropped_import_count = max(0, raw_import_count - len(df_csv))
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
                if dropped_import_count > 0:
                    st.sidebar.warning(f"有 {dropped_import_count} 行未导入：通常是 BV 格式缺失/不完整。B站 BV 号一般应为 BV + 10 位字符，例如 BV1xx411c7mD。")
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


# =========================
# KOL 对比评分 + 可视化
# =========================
KOL_GRADE_ORDER = ["A 重点续约", "B 可继续", "C 待观察", "D 谨慎投放", "E 无法判断"]
KOL_GRADE_COLORS = {
    "A 重点续约": "#2ca02c",   # 绿色：明确优先
    "B 可继续": "#1f77b4",     # 蓝色：继续合作
    "C 待观察": "#ff7f0e",     # 橙色：观察验证
    "D 谨慎投放": "#d62728",   # 红色：风险/收缩
    "E 无法判断": "#7f7f7f",   # 灰色：无基准
}


def _lift_value(current: float, baseline: float) -> float:
    try:
        current = float(current)
        baseline = float(baseline)
        if np.isnan(current) or np.isnan(baseline) or baseline <= 0:
            return np.nan
        return current / baseline - 1.0
    except Exception:
        return np.nan


def _fmt_lift(x) -> str:
    try:
        x = float(x)
        if np.isnan(x):
            return "-"
        return f"{x*100:.1f}%"
    except Exception:
        return "-"


def _fmt_rate(x) -> str:
    try:
        x = float(x)
        if np.isnan(x):
            return "-"
        return f"{x*100:.2f}%"
    except Exception:
        return "-"


def _score_component(lift: float, scale: float) -> float:
    """把提升值映射到 0-100；0%提升约等于 50 分，避免极端播放提升无限拉高。"""
    try:
        lift = float(lift)
        if np.isnan(lift):
            return 50.0
        return float(np.clip(50 + 45 * np.tanh(lift / scale), 0, 100))
    except Exception:
        return 50.0


def _grade_from_score(score: float) -> str:
    try:
        score = float(score)
        if np.isnan(score):
            return "E 无法判断"
    except Exception:
        return "E 无法判断"
    if score >= 80:
        return "A 重点续约"
    if score >= 65:
        return "B 可继续"
    if score >= 50:
        return "C 待观察"
    return "D 谨慎投放"


def _recommendation(view_lift: float, er_lift: float, deep_lift: float, base_type: str, sample_n: int, min_n: int) -> tuple[str, float]:
    """推荐等级完全由最终分数决定，保证颜色、评级、分数三者一致。"""
    if base_type == "无基准":
        return "E 无法判断", np.nan

    view_score = _score_component(view_lift, 1.20)
    er_score = _score_component(er_lift, 0.70)
    deep_score = _score_component(deep_lift, 0.80)
    score = 0.50 * view_score + 0.35 * er_score + 0.15 * deep_score

    # 质量约束：播放爆量但互动明显下滑，不允许进入高等级。
    if not np.isnan(er_lift) and er_lift <= -0.25:
        score = min(score, 59.0)
    if not np.isnan(view_lift) and view_lift <= -0.35:
        score = min(score, 59.0)
    if (not np.isnan(er_lift) and er_lift <= -0.45) and (not np.isnan(deep_lift) and deep_lift <= -0.35):
        score = min(score, 49.0)

    # 样本可靠性约束：小样本/替代基准不能直接给 A。
    if base_type == "平台替代基准":
        score = min(score, 69.0)
    elif sample_n < min_n:
        score = min(score, 74.0)

    score = float(np.clip(score, 0, 100))
    return _grade_from_score(score), score


def _baseline_reliability(base_type: str, sample_n: int, min_n: int) -> str:
    if base_type == "KOL历史基准" and sample_n >= min_n:
        return "高：KOL历史足量"
    if base_type == "KOL历史小样本":
        return f"中：KOL历史小样本({sample_n}/{min_n})"
    if base_type == "平台替代基准":
        return "低：平台替代基准"
    return "无：无法判断"


def _build_kol_compare_lib(
    df_all_m: pd.DataFrame,
    collab_projects: list[str],
    baseline_window_n: int,
    baseline_min_n: int,
    name_map: dict | None = None,
    use_proxy_baseline: bool = True,
) -> pd.DataFrame:
    """生成 KOL 对比库；个人基准不足时用明确标注的平台替代基准兜底，不让汇报页空白。"""
    if df_all_m is None or df_all_m.empty:
        return pd.DataFrame()
    df_all_m = compute_metrics(df_all_m.copy())
    df_all_m["owner_mid"] = df_all_m["owner_mid"].apply(_norm_mid)
    collab_set = set(collab_projects or [])
    name_map = name_map or {}

    collab_mid_df = df_all_m[df_all_m["project"].isin(collab_set)].copy()
    collab_mid_df = collab_mid_df[collab_mid_df["owner_mid"].astype(str).str.len() > 0]
    if collab_mid_df.empty:
        return pd.DataFrame()

    # 平台替代基准：全库中非本次合作项目的视频，排除明显无效播放。
    proxy_pool = df_all_m[(~df_all_m["project"].isin(collab_set)) | (df_all_m["project"] == BASELINE_PROJECT)].copy()
    proxy_pool = proxy_pool[proxy_pool["view"].astype(float) > 0].drop_duplicates(subset=["owner_mid", "bvid"], keep="last")
    proxy_pool = _sort_owner_hist(proxy_pool).head(max(baseline_window_n * 20, 120)) if not proxy_pool.empty else proxy_pool

    rows = []
    for mid, g_collab in collab_mid_df.groupby("owner_mid"):
        up_name = (g_collab["owner_name"].value_counts().index[0]
                   if not g_collab["owner_name"].dropna().empty else name_map.get(mid, ""))
        owner_all = df_all_m[df_all_m["owner_mid"] == mid].copy()
        owner_all = _sort_owner_hist(owner_all)
        collab_bvids = set(g_collab["bvid"].astype(str).tolist())

        own_base = owner_all[(~owner_all["project"].isin(collab_set)) | (owner_all["project"] == BASELINE_PROJECT)].copy()
        own_base = own_base[~own_base["bvid"].astype(str).isin(collab_bvids)]
        own_base = own_base[own_base["view"].astype(float) > 0]
        own_base = own_base.drop_duplicates(subset=["bvid"], keep="last")
        own_base = _sort_owner_hist(own_base).head(baseline_window_n)

        if len(own_base) >= baseline_min_n:
            base_pool = own_base
            base_type = "KOL历史基准"
        elif len(own_base) > 0:
            base_pool = own_base
            base_type = "KOL历史小样本"
        elif use_proxy_baseline and proxy_pool is not None and not proxy_pool.empty:
            base_pool = proxy_pool.head(baseline_window_n)
            base_type = "平台替代基准"
        else:
            base_pool = pd.DataFrame()
            base_type = "无基准"

        collab_view = float(g_collab["view"].median()) if not g_collab.empty else np.nan
        collab_er = float(g_collab["engagement_rate"].median()) if not g_collab.empty else np.nan
        collab_deep = float(g_collab["deep_signal_ratio"].median()) if not g_collab.empty else np.nan

        if base_pool.empty:
            base_view = base_er = base_deep = np.nan
        else:
            base_view = float(base_pool["view"].median())
            base_er = float(base_pool["engagement_rate"].median())
            base_deep = float(base_pool["deep_signal_ratio"].median())

        view_lift = _lift_value(collab_view, base_view)
        er_lift = _lift_value(collab_er, base_er)
        deep_lift = _lift_value(collab_deep, base_deep)
        rec, score = _recommendation(view_lift, er_lift, deep_lift, base_type, len(base_pool), baseline_min_n)
        reliability = _baseline_reliability(base_type, len(base_pool), baseline_min_n)

        mark = kol_flag(view_lift, er_lift, deep_lift)
        if base_type == "平台替代基准":
            mark = "🟡 替代基准待验证"
        elif base_type == "KOL历史小样本":
            mark = mark or "🔎 小样本可读"
        elif base_type == "无基准":
            mark = "⚪ 无基准"

        tags = []
        if not np.isnan(view_lift) and view_lift >= 0.30: tags.append("热度拉升")
        if not np.isnan(er_lift) and er_lift >= 0.20: tags.append("互动增强")
        if not np.isnan(deep_lift) and deep_lift >= 0.10: tags.append("沉淀提升")
        if not tags: tags.append("常规")
        persona = f"{'热度拉升' if '热度拉升' in tags else '热度稳定'} + {'互动增强' if '互动增强' in tags else '互动常规'} + {'沉淀提升' if '沉淀提升' in tags else '沉淀一般'}"

        rows.append({
            "owner_mid": mid,
            "KOL/UP主": up_name,
            "推荐等级": rec,
            "综合评分": score,
            "标注": mark,
            "合作视频数": int(len(g_collab)),
            "基准样本数": int(len(base_pool)),
            "基准类型": base_type,
            "基准可靠性": reliability,
            "标签": "、".join(tags),
            "KOL画像一句话": persona,
            "合作播放中位数": int(collab_view) if not np.isnan(collab_view) else 0,
            "基准播放中位数": int(base_view) if not np.isnan(base_view) else 0,
            "播放提升值": view_lift,
            "播放提升": _fmt_lift(view_lift),
            "合作互动率值": collab_er,
            "基准互动率值": base_er,
            "合作互动率中位数": _fmt_rate(collab_er),
            "基准互动率中位数": _fmt_rate(base_er),
            "互动率提升值": er_lift,
            "互动率提升": _fmt_lift(er_lift),
            "合作深度信号值": collab_deep,
            "基准深度信号值": base_deep,
            "合作深度信号中位数": _fmt_rate(collab_deep),
            "基准深度信号中位数": _fmt_rate(base_deep),
            "深度信号提升值": deep_lift,
            "深度信号提升": _fmt_lift(deep_lift),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["推荐等级"] = pd.Categorical(out["推荐等级"], categories=KOL_GRADE_ORDER, ordered=True)
    return out.sort_values(["推荐等级", "综合评分"], ascending=[True, False]).reset_index(drop=True)


def _render_kol_visuals(lib: pd.DataFrame):
    """KOL视觉总览：颜色只代表推荐等级；基准可靠性用符号/悬浮说明，避免颜色混淆。"""
    if lib is None or lib.empty:
        st.info("暂无KOL结果可视化。")
        return
    d = lib.copy()
    for c in ["播放提升值", "互动率提升值", "深度信号提升值", "综合评分", "合作播放中位数"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["推荐等级"] = d["推荐等级"].astype(str)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KOL总数", f"{len(d):,}")
    c2.metric("A重点续约", f"{int((d['推荐等级'] == 'A 重点续约').sum()):,}")
    c3.metric("B可继续", f"{int((d['推荐等级'] == 'B 可继续').sum()):,}")
    c4.metric("C/D观察或收缩", f"{int(d['推荐等级'].isin(['C 待观察','D 谨慎投放']).sum()):,}")

    st.markdown("**KOL投放四象限（颜色=推荐等级；符号=基准可靠性；只标注重点/异常账号）**")
    plot_df = d.dropna(subset=["播放提升值", "互动率提升值"]).copy()
    if plot_df.empty:
        st.info("当前KOL缺少可绘制的提升值。")
    else:
        plot_df["播放提升显示"] = plot_df["播放提升值"].clip(-1.0, 3.0)
        plot_df["互动率提升显示"] = plot_df["互动率提升值"].clip(-1.0, 2.0)
        plot_df["气泡播放"] = np.log10(plot_df["合作播放中位数"].clip(lower=1))
        plot_df["基准可靠性简写"] = plot_df["基准可靠性"].astype(str).str.extract(r"^(高|中|低|无)", expand=False).fillna("无")
        plot_df = plot_df.sort_values("综合评分", ascending=False).reset_index(drop=True)
        # 只标注：A、D、播放体量Top、提升异常Top，最多 12 个，避免挤成一团。
        label_idx = set(plot_df[plot_df["推荐等级"].isin(["A 重点续约", "D 谨慎投放"])].index.tolist())
        label_idx.update(plot_df.nlargest(min(5, len(plot_df)), "合作播放中位数").index.tolist())
        label_idx.update(plot_df.assign(_dist=(plot_df["播放提升显示"].abs() + plot_df["互动率提升显示"].abs())).nlargest(min(5, len(plot_df)), "_dist").index.tolist())
        label_idx = set(list(label_idx)[:12])
        plot_df["显示标签"] = [name if i in label_idx else "" for i, name in enumerate(plot_df["KOL/UP主"].astype(str).tolist())]

        fig = px.scatter(
            plot_df,
            x="播放提升显示",
            y="互动率提升显示",
            size="气泡播放",
            color="推荐等级",
            symbol="基准可靠性简写",
            text="显示标签",
            color_discrete_map=KOL_GRADE_COLORS,
            category_orders={"推荐等级": KOL_GRADE_ORDER, "基准可靠性简写": ["高", "中", "低", "无"]},
            hover_name="KOL/UP主",
            hover_data={
                "播放提升显示": False,
                "互动率提升显示": False,
                "播放提升": True,
                "互动率提升": True,
                "深度信号提升": True,
                "综合评分": ":.1f",
                "推荐等级": True,
                "基准可靠性": True,
                "基准类型": True,
                "合作播放中位数": ":,",
                "基准播放中位数": ":,",
                "气泡播放": False,
                "显示标签": False,
            },
            height=660,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vrect(x0=0, x1=3, y0=0, y1=2, fillcolor="#2ca02c", opacity=0.06, line_width=0)
        fig.add_annotation(x=2.65, y=1.75, text="高播放 + 高互动<br>优先续投", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=2.65, y=-0.78, text="有播放但互动弱<br>谨慎放量", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=-0.82, y=1.72, text="互动好但播放弱<br>看素材匹配", showarrow=False, font=dict(size=13))
        fig.add_annotation(x=-0.82, y=-0.78, text="播放/互动双弱<br>减少投放", showarrow=False, font=dict(size=13))
        fig.update_traces(textposition="top center", marker=dict(opacity=0.82, line=dict(width=1, color="white")))
        fig.update_layout(
            xaxis_title="播放提升（显示截断：-100% ~ +300%，真实值看悬浮）",
            yaxis_title="互动率提升（显示截断：-100% ~ +200%，真实值看悬浮）",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            legend_title_text="推荐等级 / 基准可靠性",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("颜色只代表推荐等级：A绿、B蓝、C橙、D红、E灰；基准可靠性只用点形区分，不再参与颜色编码。")

    st.markdown("**KOL综合评分排行（0-100分；颜色与推荐等级完全一致）**")
    top_rank = d.dropna(subset=["综合评分"]).sort_values("综合评分", ascending=False).head(25).copy()
    if top_rank.empty:
        st.info("暂无可排名KOL。")
    else:
        fig_rank = px.bar(
            top_rank.sort_values("综合评分", ascending=True),
            x="综合评分",
            y="KOL/UP主",
            orientation="h",
            color="推荐等级",
            color_discrete_map=KOL_GRADE_COLORS,
            category_orders={"推荐等级": KOL_GRADE_ORDER},
            text="综合评分",
            hover_data=["基准可靠性", "播放提升", "互动率提升", "深度信号提升", "合作播放中位数", "基准播放中位数"],
            height=max(440, min(820, 30 * len(top_rank) + 160)),
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rank.update_layout(xaxis_range=[0, 105], legend_title_text="推荐等级", yaxis_title="KOL/UP主")
        st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("**投放建议分层（汇报摘要）**")
    cols = st.columns(4)
    for col, grade, title in zip(cols, KOL_GRADE_ORDER[:4], ["A 重点续约", "B 可继续", "C 待观察", "D 谨慎投放"]):
        names = d[d["推荐等级"] == grade].sort_values("综合评分", ascending=False)["KOL/UP主"].head(8).tolist()
        col.markdown(f"**{title}**")
        col.write("、".join(names) if names else "暂无")

# =========================================================
# KOL module（按 owner_mid，对齐+补齐+标注+导出）
# =========================================================
st.divider()
st.subheader("KOL合作资料库（独立模块：标注合作是否优于平时｜按owner_mid对齐）")

with st.expander("KOL模块设置", expanded=False):
    collab_projects = st.multiselect("哪些项目算合作项目", projects, default=sel_projects if sel_projects else projects)
    fetch_n = st.slider("补齐基准：每个KOL抓取最近N条公开视频", 10, 80, 30, step=5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)
    use_browser_fallback = st.checkbox("启用 Selenium 无头浏览器兜底（本机需安装 Chrome/Driver；一般不用开）", value=False)
    show_kol_quality_hint = st.checkbox("显示数据质量提示（缺mid/异常mid/抓取诊断）", value=True)
    use_proxy_baseline = st.checkbox("无个人历史基准时，生成表使用平台替代基准（会明确标注）", value=True)
    bili_cookie = st.text_input(
        "B站 Cookie（可选，仅KOL抓取用；公开数据通常留空即可）",
        value=os.environ.get("BILI_COOKIE", ""),
        type="password",
        help="如遇接口返回空或风控，可临时粘贴浏览器 Cookie；不要把 Cookie 写进代码或上传到公共仓库。"
    )
    bili_proxy = st.text_input(
        "代理地址（可选；本机Clash常见为 http://127.0.0.1:7890，Streamlit Cloud通常留空）",
        value=os.environ.get("BILI_PROXY", ""),
        help="浏览器开VPN不等于Streamlit/Python请求也走VPN；本地运行时可填代理地址，云端部署通常不能使用你本机代理。"
    )

cA, cB, cC = st.columns([1, 1, 2])
with cA:
    btn_fill_all = st.button("🧲 一键补齐所有合作KOL基准（写入__BASELINE__）")
with cB:
    btn_build_kol = st.button("📚 生成KOL对比表（含视觉总览）")
with cC:
    st.caption("抓取链路：先修复合作BV owner_mid，再聚合 Web WBI / 老接口 / APP cursor / 合集 / 动态 / 昵称搜索 / HTML / Selenium。颜色口径：A绿、B蓝、C橙、D红、E灰。")

if collab_projects:
    collab_df = df_db[df_db["project"].isin(collab_projects)].copy()
    collab_df["owner_mid"] = collab_df["owner_mid"].apply(_norm_mid)

    valid_mid_df = collab_df[collab_df["owner_mid"].astype(str).str.len() > 0].copy()
    invalid_mid_cnt = int((collab_df["owner_mid"].astype(str).str.len() == 0).sum())

    st.caption(
        f"合作UP主数：{valid_mid_df['owner_mid'].nunique()}（可识别mid）"
        f"｜缺/异常mid合作视频：{invalid_mid_cnt}"
        f"｜合作视频数：{len(collab_df)}"
    )

    if show_kol_quality_hint:
        bad_rows = collab_df[collab_df["owner_mid"].astype(str).str.len() == 0].copy()
        if not bad_rows.empty:
            st.warning(f"发现 {len(bad_rows)} 条合作视频 owner_mid 缺失/异常。点击“一键补齐”时会先用 BV 详情接口自动修复。")
            st.dataframe(bad_rows[["project","bvid","title","owner_name","owner_mid"]].head(80), use_container_width=True, height=220)
        else:
            st.success("合作视频 owner_mid 看起来正常。")
        if st.session_state.get("kol_last_debug"):
            with st.expander("上次KOL抓取诊断日志（用于定位到底卡在哪个接口）", expanded=False):
                dbg_prev = pd.DataFrame(st.session_state.get("kol_last_debug", []))
                if not dbg_prev.empty:
                    st.dataframe(dbg_prev, use_container_width=True, height=300)
                    fail_prev = dbg_prev[dbg_prev["ok"].astype(str).isin(["False", "false", "0"])] if "ok" in dbg_prev.columns else pd.DataFrame()
                    if not fail_prev.empty and "source" in fail_prev.columns:
                        st.write("失败/空结果来源统计：")
                        st.dataframe(fail_prev.groupby(["source", "code"]).size().reset_index(name="次数").sort_values("次数", ascending=False), use_container_width=True, height=220)

    name_map = (valid_mid_df.groupby("owner_mid")["owner_name"]
                .agg(lambda s: s.value_counts().index[0]).to_dict()) if not valid_mid_df.empty else {}

    if btn_fill_all:
        progress = st.progress(0)
        status_box = st.empty()
        debug_rows = []
        rows_to_write = {}
        stat = {
            "collab_repair_total": 0, "collab_repair_ok": 0, "collab_repair_fail": 0,
            "list_fail": 0, "list_empty": 0, "detail_ok": 0, "detail_fail": 0, "vlist_added": 0,
        }

        # A0. 先刷新合作BV详情，修复 CSV 导入缺 owner_mid 的根因
        collab_pairs_all = (df_db[(df_db["project"].isin(collab_projects)) & (df_db["project"] != BASELINE_PROJECT)]
                            [["project","bvid","url"]]
                            .dropna(subset=["project","bvid"])
                            .drop_duplicates()
                            .values
                            .tolist())
        stat["collab_repair_total"] = len(collab_pairs_all)
        detail_sess = _make_bili_session(cookie=bili_cookie, proxy=bili_proxy)
        total_steps = max(1, len(collab_pairs_all) + max(1, valid_mid_df["owner_mid"].nunique()) * 2)
        step = 0

        for proj, bvid, url in collab_pairs_all:
            step += 1
            progress.progress(min(1.0, step / total_steps))
            status_box.info(f"正在刷新合作BV详情/修复 owner_mid：{bvid}")
            bvid = str(bvid)
            if not bvid.startswith("BV"):
                stat["collab_repair_fail"] += 1
                continue
            detail = fetch_video_detail_by_bvid(bvid, sess=detail_sess)
            if detail is None:
                stat["collab_repair_fail"] += 1
                debug_rows.append({"owner_mid": "", "KOL/UP主": "", "source": "collab_detail_repair", "ok": False, "count": 0, "code": "DETAIL_FAIL", "message": bvid, "time": pd.Timestamp.now().strftime("%H:%M:%S")})
                _sleep_jitter(float(sleep_sec))
                continue
            detail_row = _detail_row_for_project(detail, project=str(proj), url=_safe_str(url), data_type="collab", baseline_for="")
            rows_to_write[(str(proj), bvid)] = detail_row
            stat["collab_repair_ok"] += 1
            _sleep_jitter(float(sleep_sec))

        df_db_work = df_db.copy()
        if rows_to_write:
            df_rep = normalize_df(pd.DataFrame(list(rows_to_write.values())))
            df_rep["pubdate"] = pd.to_datetime(df_rep["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_rep["fetched_at"] = pd.to_datetime(df_rep["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_rep)
            df_db_work = normalize_df(load_all_rows())

        # A1. 用修复后的 owner_mid 重新抓个人基准
        collab_df_work = df_db_work[df_db_work["project"].isin(collab_projects)].copy()
        collab_df_work["owner_mid"] = collab_df_work["owner_mid"].apply(_norm_mid)
        valid_mid_df_work = collab_df_work[collab_df_work["owner_mid"].astype(str).str.len() > 0].copy()
        name_map_work = (valid_mid_df_work.groupby("owner_mid")["owner_name"]
                         .agg(lambda s: s.value_counts().index[0]).to_dict()) if not valid_mid_df_work.empty else {}
        collab_bvids_all = set(valid_mid_df_work["bvid"].astype(str).tolist())
        baseline_rows_to_write = {}

        mids = sorted(valid_mid_df_work["owner_mid"].unique().tolist())
        if not mids:
            st.error("合作视频详情刷新后仍没有可用 owner_mid：请检查 BV 是否有效，或稍后重试详情接口。")
        else:
            for mid in mids:
                step += 1
                progress.progress(min(1.0, step / total_steps))
                disp = name_map_work.get(mid, "")
                status_box.info(f"正在抓取KOL基准：{disp or mid}（mid={mid}）")
                per_mid_debug = []
                try:
                    vlist = fetch_vlist_by_mid(
                        int(mid),
                        n=int(fetch_n),
                        use_browser_fallback=bool(use_browser_fallback),
                        sleep_sec=float(sleep_sec),
                        cookie=bili_cookie,
                        proxy=bili_proxy,
                        owner_name=disp,
                        debug=per_mid_debug,
                    )
                    for d in per_mid_debug:
                        d["owner_mid"] = mid
                        d["KOL/UP主"] = disp
                        debug_rows.append(d)
                except Exception as e:
                    stat["list_fail"] += 1
                    debug_rows.append({"owner_mid": mid, "KOL/UP主": disp, "source": "fetch_vlist_by_mid", "ok": False, "count": 0, "code": "EXC", "message": str(e), "time": pd.Timestamp.now().strftime("%H:%M:%S")})
                    continue

                if not vlist:
                    stat["list_empty"] += 1
                    continue

                for v in vlist:
                    bvid = v.get("bvid", "")
                    if not bvid or bvid in collab_bvids_all:
                        continue
                    base_row = {
                        "project": BASELINE_PROJECT,
                        "bvid": bvid,
                        "url": f"https://www.bilibili.com/video/{bvid}",
                        "title": v.get("title", ""),
                        "pubdate": v.get("pubdate", pd.NaT),
                        "owner_mid": mid,
                        "owner_name": disp,
                        "view": _safe_int(v.get("view", 0)),
                        "reply": _safe_int(v.get("reply", 0)),
                        "like": _safe_int(v.get("like", 0)),
                        "coin": _safe_int(v.get("coin", 0)),
                        "favorite": _safe_int(v.get("favorite", 0)),
                        "danmaku": _safe_int(v.get("danmaku", 0)),
                        "share": _safe_int(v.get("share", 0)),
                        "fans_delta": 0,
                        "baseline_for": disp,
                        "data_type": "baseline",
                        "fetched_at": pd.Timestamp.now(),
                    }
                    baseline_rows_to_write[(BASELINE_PROJECT, bvid)] = base_row
                    stat["vlist_added"] += 1

                    detail = fetch_video_detail_by_bvid(bvid, sess=detail_sess)
                    if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                        detail_row = _detail_row_for_project(detail, project=BASELINE_PROJECT, url=f"https://www.bilibili.com/video/{bvid}", data_type="baseline", baseline_for=disp)
                        if not detail_row.get("owner_name"):
                            detail_row["owner_name"] = disp
                        baseline_rows_to_write[(BASELINE_PROJECT, bvid)] = detail_row
                        stat["detail_ok"] += 1
                    else:
                        stat["detail_fail"] += 1
                    _sleep_jitter(float(sleep_sec))

        if baseline_rows_to_write:
            df_new = normalize_df(pd.DataFrame(list(baseline_rows_to_write.values())))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        progress.progress(1.0)
        status_box.success("KOL补齐流程完成。")
        st.session_state["kol_last_debug"] = debug_rows

        if debug_rows and show_kol_quality_hint:
            with st.expander("本次KOL抓取诊断日志", expanded=True):
                dbg = pd.DataFrame(debug_rows)
                st.dataframe(dbg, use_container_width=True, height=320)
                fail_dbg = dbg[dbg["ok"] == False] if "ok" in dbg.columns else pd.DataFrame()
                if not fail_dbg.empty:
                    st.write("失败/空结果来源统计：")
                    st.dataframe(fail_dbg.groupby(["source", "code"]).size().reset_index(name="次数").sort_values("次数", ascending=False), use_container_width=True, height=220)

        if baseline_rows_to_write or rows_to_write:
            st.success(
                f"完成：合作BV详情刷新 {stat['collab_repair_ok']}/{stat['collab_repair_total']}；"
                f"新增/覆盖 {stat['vlist_added']} 条基准；"
                f"列表失败 {stat['list_fail']}，列表空 {stat['list_empty']}；"
                f"详情补全成功 {stat['detail_ok']}，失败 {stat['detail_fail']}。"
                f"（数据已自动备份到 backup/backup_latest.csv）"
            )
            st.rerun()
        else:
            st.warning("本次未写入任何数据：合作BV详情、KOL列表、基准详情都未成功。请展开诊断日志看具体接口返回；必要时填写 B站 Cookie 或改用本地运行+代理。")

    st.markdown("**KOL基准诊断（按owner_mid统计库内数量）**")
    diag = []
    for mid in sorted(valid_mid_df["owner_mid"].unique().tolist()):
        owner_all = df_db[df_db["owner_mid"].apply(_norm_mid) == mid].copy()
        owner_all = _sort_owner_hist(owner_all)
        collab_bvids = set(collab_df[collab_df["owner_mid"] == mid]["bvid"].astype(str).tolist())
        base_pool = owner_all[(~owner_all["project"].isin(set(collab_projects))) | (owner_all["project"] == BASELINE_PROJECT)].copy()
        base_pool = base_pool[~base_pool["bvid"].astype(str).isin(collab_bvids)]
        base_pool = base_pool[base_pool["view"].astype(float) > 0]
        base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")
        base_pool = _sort_owner_hist(base_pool)
        avail = int(min(len(base_pool), baseline_window_n))
        if avail >= baseline_min_n:
            status_text = "OK"
            advice = "个人历史足量，可直接比较"
        elif avail > 0:
            status_text = f"小样本({avail}/{baseline_min_n})"
            advice = "可参考，但建议继续补齐"
        else:
            status_text = "无个人历史基准"
            advice = "将继续尝试多接口；生成表可使用平台替代基准"
        diag.append({
            "owner_mid": mid,
            "KOL/UP主": name_map.get(mid, owner_all["owner_name"].dropna().iloc[0] if not owner_all.empty else ""),
            "库内视频总数": int(len(owner_all)),
            "可用个人基准数": int(len(base_pool)),
            f"取最近{baseline_window_n}可用": avail,
            "状态": status_text,
            "建议": advice,
        })
    if diag:
        diag_df = pd.DataFrame(diag).sort_values(["状态", "可用个人基准数"], ascending=[True, False])
        st.dataframe(diag_df, use_container_width=True, height=360)
        no_base_cnt = int((diag_df["可用个人基准数"] == 0).sum())
        if no_base_cnt > 0:
            st.info(f"还有 {no_base_cnt} 个KOL没有个人历史基准。通常卡在：UP隐藏投稿/公开视频很少、空间接口风控、搜索召回不到该mid。生成表时会用‘平台替代基准’并在表中明确标注，不会伪装成个人历史基准。")
    else:
        st.info("暂无可诊断的 owner_mid。点击“一键补齐”会先尝试用合作BV详情修复 owner_mid。")

    if btn_build_kol:
        df_all_m = normalize_df(load_all_rows())
        lib = _build_kol_compare_lib(
            df_all_m=df_all_m,
            collab_projects=collab_projects,
            baseline_window_n=baseline_window_n,
            baseline_min_n=baseline_min_n,
            name_map=name_map,
            use_proxy_baseline=bool(use_proxy_baseline),
        )
        if lib.empty:
            st.warning("没有生成KOL结果：请先补齐基准，或检查合作项目是否包含有效 owner_mid。")
        else:
            tab_visual, tab_table = st.tabs(["📍 KOL视觉总览", "📋 KOL对比表校对"])
            with tab_visual:
                _render_kol_visuals(lib)
            with tab_table:
                display_cols = [
                    "owner_mid", "KOL/UP主", "推荐等级", "综合评分", "标注", "合作视频数", "基准样本数", "基准类型", "基准可靠性",
                    "标签", "KOL画像一句话", "合作播放中位数", "基准播放中位数", "播放提升",
                    "合作互动率中位数", "基准互动率中位数", "互动率提升",
                    "合作深度信号中位数", "基准深度信号中位数", "深度信号提升"
                ]
                show_lib = lib.copy()
                if "综合评分" in show_lib.columns:
                    show_lib["综合评分"] = pd.to_numeric(show_lib["综合评分"], errors="coerce").round(1)
                st.dataframe(show_lib[[c for c in display_cols if c in show_lib.columns]], use_container_width=True, height=520)
                st.download_button(
                    "⬇️ 下载KOL对比表（CSV）",
                    data=show_lib.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kol_compare.csv",
                    mime="text/csv"
                )
else:
    st.info("请先在 KOL模块设置 中选择合作项目。")
