
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
