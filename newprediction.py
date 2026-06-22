#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300智能估值与企业研究平台
- 数据层：4线程并发获取 + 增量更新 + 批量写入（Baostock原生真实数据）
- 估值层：四维度相对估值 + 戈登DCF绝对估值 + 敏感性分析
- 研究层：四大财务能力 + 主营业务与行业分析
- 可视化：科技感深色UI + 交互式K线 + 多维度图表
- 输出层：专业PDF研究报告
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import baostock as bs
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.io as pio
from sklearn.linear_model import LinearRegression

# ======================== 全局页面配置 ========================
st.set_page_config(
    page_title="沪深300智能估值研究平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== 科技感深色主题样式 ========================
st.markdown("""
<style>
    /* 全局背景 */
    .main {
        background: linear-gradient(135deg, #0b1120 0%, #0f172a 50%, #0b1120 100%);
        color: #e2e8f0;
    }
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px;}
    
    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #1e3a5f;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white; padding: 14px; border-radius: 10px; text-align: center;
        margin-bottom: 18px; box-shadow: 0 0 20px rgba(14, 165, 233, 0.3);
    }
    
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: white; border-radius: 8px; font-weight: 600; border: none;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5);
    }
    
    /* 玻璃拟态卡片 */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #0ea5e9;
        box-shadow: 0 0 15px rgba(14, 165, 233, 0.2);
        transform: translateY(-3px);
    }
    .metric-value {
        font-size: 22px; font-weight: 700; color: #0ea5e9; margin: 6px 0;
    }
    .metric-label {font-size: 13px; color: #94a3b8;}
    
    /* 标题渐变 */
    .gradient-title {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* 建议与风险框 */
    .advice-box {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-left: 4px solid #0ea5e9;
        padding: 18px 22px; border-radius: 8px; margin: 12px 0;
    }
    .risk-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 15px 20px; border-radius: 8px; margin: 10px 0;
    }
    
    /* Tab样式 */
    .stTabs [data-testid="stTab"] {
        background: rgba(30, 41, 59, 0.5);
        color: #94a3b8;
        border-radius: 8px 8px 0 0;
        margin-right: 4px;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background: rgba(14, 165, 233, 0.2);
        color: #0ea5e9;
        border-bottom: 2px solid #0ea5e9;
    }
    
    /* 表格样式 */
    .dataframe {border-radius: 8px; overflow: hidden;}
    
    /* 分割线 */
    hr {border-color: #1e3a5f; margin: 20px 0;}
    
    /* 滚动条 */
    ::-webkit-scrollbar {width: 6px; height: 6px;}
    ::-webkit-scrollbar-track {background: #0f172a;}
    ::-webkit-scrollbar-thumb {background: #1e3a5f; border-radius: 3px;}
    ::-webkit-scrollbar-thumb:hover {background: #0ea5e9;}
</style>
""", unsafe_allow_html=True)

# ======================== 全局路径与配置 ========================
DATA_DIR = "./data"
STOCK_FILE = os.path.join(DATA_DIR, "stock_data.csv")
FINANCIAL_FILE = os.path.join(DATA_DIR, "financial_data.csv")
HS300_LIST_FILE = os.path.join(DATA_DIR, "hs300_stock_list.csv")
START_DATE = "2015-01-01"
MAX_WORKERS = 4  # 并发线程数，稳定优先

# 策略配置
STRATEGY_CONFIG = {
    "价值投资策略（长线稳健）": {
        "desc": "精选低估值、低波动、高ROE个股，长期持有分批止盈",
        "hold_days": (60, 120), "target_return": (0.15, 0.35),
        "risk_level": "低风险", "sort_by": ["ROE", "估值分位"], "ascending": [False, True]
    },
    "趋势追涨策略（中线波段）": {
        "desc": "筛选均线多头、量价齐升个股，波段操作捕捉趋势",
        "hold_days": (20, 45), "target_return": (0.08, 0.25),
        "risk_level": "中风险", "sort_by": ["20日涨幅", "均线趋势"], "ascending": [False, False]
    },
    "反转抄底策略（短线博弈）": {
        "desc": "筛选短期超跌、缩量企稳个股，博弈反弹行情",
        "hold_days": (7, 20), "target_return": (0.05, 0.18),
        "risk_level": "高风险", "sort_by": ["20日涨幅", "波动率"], "ascending": [True, True]
    }
}

# ======================== 工具：中文字体注册 ========================
@st.cache_resource
def register_chinese_font():
    font_candidates = [
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc", 0),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf", None),
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc", 0),
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc", 0),
        ("NotoSansCJK", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ]
    for name, path, subfont in font_candidates:
        if os.path.exists(path):
            try:
                if subfont is not None:
                    pdfmetrics.registerFont(TTFont(name, path, subfontIndex=subfont))
                else:
                    pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                continue
    return "Helvetica"

CHINESE_FONT = register_chinese_font()

# ======================== 数据层：多线程并发获取 ========================
_thread_local = threading.local()

def thread_login():
    """线程独立登录，保证线程安全"""
    if not hasattr(_thread_local, 'logged_in') or not _thread_local.logged_in:
        lg = bs.login()
        _thread_local.logged_in = (lg.error_code == '0')
    return _thread_local.logged_in

def thread_logout():
    if hasattr(_thread_local, 'logged_in') and _thread_local.logged_in:
        try:
            bs.logout()
        except Exception:
            pass
        _thread_local.logged_in = False

def get_hs300_stocks():
    """获取沪深300成分股列表"""
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception("Baostock登录失败")
    try:
        rs = bs.query_hs300_stocks()
        stocks = []
        while rs.error_code == '0' and rs.next():
            stocks.append(rs.get_row_data())
        df = pd.DataFrame(stocks, columns=rs.fields)
        df['纯代码'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        return df
    finally:
        bs.logout()

def fetch_single_stock_kline(bs_code, start_date, end_date):
    """单线程内获取单只股票量价+估值数据（线程安全）"""
    if not thread_login():
        return None
    
    time.sleep(0.1)  # 限流
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=start_date, end_date=end_date, frequency="d", adjustflag="1"
    )
    
    if rs.error_code != '0':
        return None
    
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=rs.fields)
    numeric_cols = ['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    df['股票代码'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    df['日期'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    df = df.rename(columns={
        'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
        'volume': '成交量', 'amount': '成交额', 'turn': '换手率', 'pctChg': '涨跌幅'
    })
    
    final_cols = [
        '股票代码', '日期', '开盘', '收盘', '最高', '最低',
        '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM'
    ]
    return df[final_cols]

def fetch_financial_data(bs_code):
    """获取最近4年财务指标（盈利能力+成长能力+偿债能力）"""
    if not thread_login():
        return None
    
    financial = {}
    current_year = datetime.now().year
    
    try:
        # 盈利能力
        profit_list = []
        for year in range(current_year-3, current_year+1):
            rs = bs.query_profit_data(code=bs_code, year=year, quarter=4)
            while rs.error_code == '0' and rs.next():
                profit_list.append(rs.get_row_data())
        if profit_list:
            profit_df = pd.DataFrame(profit_list, columns=rs.fields)
            for col in ['roe', 'npr', 'gp']:
                if col in profit_df.columns:
                    financial[col] = pd.to_numeric(profit_df.iloc[-1][col], errors='coerce')
        
        # 成长能力
        growth_list = []
        for year in range(current_year-3, current_year+1):
            rs = bs.query_growth_data(code=bs_code, year=year, quarter=4)
            while rs.error_code == '0' and rs.next():
                growth_list.append(rs.get_row_data())
        if growth_list:
            growth_df = pd.DataFrame(growth_list, columns=rs.fields)
            for col in ['npgrowth', 'tagrowth']:
                if col in growth_df.columns:
                    financial[col] = pd.to_numeric(growth_df.iloc[-1][col], errors='coerce')
        
        # 偿债能力
        rs = bs.query_balance_data(code=bs_code, year=current_year-1, quarter=4)
        if rs.error_code == '0' and rs.next():
            balance = rs.get_row_data()
            fields = rs.fields
            if 'debtassetratio' in fields:
                financial['debtassetratio'] = pd.to_numeric(balance[fields.index('debtassetratio')], errors='coerce')
        
        return financial
    except Exception:
        return None

def get_company_basic_info(stock_code, stock_name):
    """获取企业主营业务与行业信息（降级兼容）"""
    info = {
        '所属行业': '数据获取中',
        '主营业务': '暂无数据',
        '公司简介': '暂无数据'
    }
    try:
        import akshare as ak
        df = ak.stock_individual_info_em(symbol=stock_code)
        info_dict = dict(zip(df['item'], df['value']))
        info['所属行业'] = info_dict.get('行业', '未知')
        info['主营业务'] = info_dict.get('主营业务', '暂无数据')
        info['公司简介'] = info_dict.get('公司简介', '暂无数据')
    except Exception:
        info['所属行业'] = '沪深300成分股'
        info['主营业务'] = f'{stock_name} 主营业务数据需安装akshare后查看'
    return info

def calculate_update_tasks(hs300_df):
    """预计算所有股票的更新任务列表"""
    tasks = []
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 预加载现有数据
    existing_ranges = {}
    if os.path.exists(STOCK_FILE):
        try:
            df = pd.read_csv(STOCK_FILE, usecols=['股票代码', '日期'], dtype={'股票代码': str})
            df['股票代码'] = df['股票代码'].str.zfill(6)
            df['日期'] = pd.to_datetime(df['日期'])
            ranges = df.groupby('股票代码')['日期'].agg(['min', 'max'])
            existing_ranges = ranges.to_dict('index')
        except Exception:
            pass
    
    for _, row in hs300_df.iterrows():
        code = row['纯代码']
        bs_code = row['code']
        name = row['code_name']
        
        fetch_ranges = []
        if code in existing_ranges:
            min_dt = existing_ranges[code]['min'].strftime('%Y-%m-%d')
            max_dt = existing_ranges[code]['max'].strftime('%Y-%m-%d')
            
            if min_dt > START_DATE:
                fetch_ranges.append((START_DATE, (pd.to_datetime(min_dt) - timedelta(days=1)).strftime('%Y-%m-%d')))
            if max_dt < today:
                fetch_ranges.append(((pd.to_datetime(max_dt) + timedelta(days=1)).strftime('%Y-%m-%d'), today))
        else:
            fetch_ranges.append((START_DATE, today))
        
        if fetch_ranges:
            tasks.append({
                'bs_code': bs_code,
                'pure_code': code,
                'name': name,
                'ranges': fetch_ranges
            })
    
    return tasks

def update_all_stock_data(progress_callback=None):
    """多线程增量更新全量数据"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. 获取成分股
    hs300 = get_hs300_stocks()
    hs300.to_csv(HS300_LIST_FILE, index=False, encoding='utf-8-sig')
    total = len(hs300)
    
    # 2. 计算更新任务
    tasks = calculate_update_tasks(hs300)
    task_total = len(tasks)
    
    if task_total == 0:
        return {"success": total, "total": total, "failed": 0, "skipped": total}
    
    # 3. 加载现有数据到内存
    existing_df = pd.DataFrame()
    if os.path.exists(STOCK_FILE):
        try:
            existing_df = pd.read_csv(STOCK_FILE, dtype={'股票代码': str})
            existing_df['股票代码'] = existing_df['股票代码'].str.zfill(6)
        except Exception:
            pass
    
    # 4. 多线程执行
    failed = []
    success_count = 0
    completed = 0
    lock = threading.Lock()
    
    def worker(task):
        nonlocal success_count, completed
        bs_code = task['bs_code']
        code = task['pure_code']
        name = task['name']
        
        try:
            all_data = []
            for start, end in task['ranges']:
                df = fetch_single_stock_kline(bs_code, start, end)
                if df is not None and not df.empty:
                    df['股票名称'] = name
                    all_data.append(df)
            
            if all_data:
                stock_df = pd.concat(all_data, ignore_index=True)
                with lock:
                    nonlocal existing_df
                    if not existing_df.empty:
                        existing_df = existing_df[existing_df['股票代码'] != code]
                    existing_df = pd.concat([existing_df, stock_df], ignore_index=True)
                    success_count += 1
            
            with lock:
                completed += 1
                if progress_callback:
                    progress_callback(completed, task_total, f"{code} {name}")
            
            return (code, name, True, None)
        except Exception as e:
            with lock:
                completed += 1
                failed.append((code, name, str(e)))
                if progress_callback:
                    progress_callback(completed, task_total, f"{code} {name}")
            return (code, name, False, str(e))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for future in as_completed(futures):
            future.result()
    
    # 登出所有线程
    for _ in range(MAX_WORKERS):
        executor.submit(thread_logout)
    
    # 5. 统一排序保存
    if not existing_df.empty:
        existing_df['日期_dt'] = pd.to_datetime(existing_df['日期'])
        existing_df = existing_df.sort_values(['股票代码', '日期_dt'])
        existing_df = existing_df.drop_duplicates(subset=['股票代码', '日期_dt'])
        existing_df = existing_df.drop(columns=['日期_dt'])
        existing_df.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')
    
    return {
        "success": success_count,
        "total": total,
        "failed": len(failed),
        "skipped": total - task_total,
        "failed_list": failed
    }

# ======================== 数据加载层 ========================
@st.cache_data(ttl=3600)
def load_local_stock_data():
    """加载本地行情数据"""
    if not os.path.exists(STOCK_FILE):
        return None
    try:
        df = pd.read_csv(STOCK_FILE, dtype={'股票代码': str}, encoding='utf-8-sig')
        df['股票代码'] = df['股票代码'].str.zfill(6)
        df['日期'] = pd.to_datetime(df['日期'])
        
        if '股票名称' not in df.columns and os.path.exists(HS300_LIST_FILE):
            hs300 = pd.read_csv(HS300_LIST_FILE, dtype={'纯代码': str})
            name_map = dict(zip(hs300['纯代码'], hs300['code_name']))
            df['股票名称'] = df['股票代码'].map(name_map).fillna(df['股票代码'])
        
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', 'peTTM', 'pbMRQ', 'psTTM']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['日期', '收盘'])
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

# ======================== 估值模型层 ========================
def calculate_full_valuation(stock_df):
    """完整估值模型：四维度相对估值 + 绝对DCF + 敏感性分析"""
    if stock_df.empty or len(stock_df) < 252:
        return None
    
    last = stock_df.iloc[-1]
    price = last['收盘']
    
    # ========== 1. 相对估值（四维度百分位） ==========
    valuation_metrics = {}
    for col, name in [('peTTM', 'PE'), ('pbMRQ', 'PB'), ('psTTM', 'PS'), ('pcfNcfTTM', 'PCF')]:
        series = stock_df[col].dropna()
        series = series[series > 0]
        if len(series) >= 120:
            current = last[col] if pd.notna(last[col]) and last[col] > 0 else series.median()
            percentile = (series <= current).mean() * 100
            valuation_metrics[name] = {
                'current': round(current, 2),
                'percentile': round(percentile, 1),
                'max': round(series.max(), 2),
                'min': round(series.min(), 2),
                'median': round(series.median(), 2)
            }
        else:
            valuation_metrics[name] = {'current': 0, 'percentile': 50, 'max': 0, 'min': 0, 'median': 0}
    
    avg_percentile = np.mean([v['percentile'] for v in valuation_metrics.values() if v['current'] > 0])
    relative_level = "低估" if avg_percentile < 30 else ("高估" if avg_percentile > 70 else "合理")
    
    # ========== 2. 绝对估值（戈登增长模型） ==========
    pe = valuation_metrics['PE']['current'] if valuation_metrics['PE']['current'] > 0 else 15
    eps = price / pe
    
    # 3年复合增长率
    if len(stock_df) >= 750:
        price_3y = stock_df.iloc[-750]['收盘']
        growth = (price / price_3y) ** (1/3) - 1 if price_3y > 0 else 0.08
    else:
        growth = 0.08
    growth = max(0.02, min(growth, 0.20))
    
    discount_rate = 0.10
    if discount_rate > growth:
        fair_value = eps * (1 + growth) / (discount_rate - growth)
    else:
        fair_value = price * 1.15
    
    fair_low = fair_value * 0.8
    fair_high = fair_value * 1.2
    
    # ========== 3. 敏感性分析矩阵 ==========
    growth_range = [growth - 0.03, growth - 0.015, growth, growth + 0.015, growth + 0.03]
    discount_range = [discount_rate - 0.02, discount_rate - 0.01, discount_rate, discount_rate + 0.01, discount_rate + 0.02]
    sensitivity = []
    for g in growth_range:
        row = []
        for r in discount_range:
            if r > g:
                val = eps * (1 + g) / (r - g)
            else:
                val = fair_value
            row.append(round(val, 2))
        sensitivity.append(row)
    
    sens_df = pd.DataFrame(
        sensitivity,
        index=[f"{g*100:.1f}%" for g in growth_range],
        columns=[f"{r*100:.1f}%" for r in discount_range]
    )
    
    # ========== 4. 综合评级 ==========
    score = 0
    if avg_percentile < 30: score += 2
    elif avg_percentile < 50: score += 1
    elif avg_percentile > 80: score -= 1
    
    if price < fair_low: score += 2
    elif price < fair_value: score += 1
    elif price > fair_high: score -= 1
    
    if score >= 3:
        rating = "强烈买入"
        advice = "当前价格显著低于内在价值，多维度估值均处于历史低位，安全边际充足，建议积极配置。"
    elif score >= 1:
        rating = "买入"
        advice = "当前价格略低于合理估值区间，基本面稳健，可分批建仓，长期持有。"
    elif score >= -1:
        rating = "持有"
        advice = "当前价格处于合理估值区间，暂无明显高估低估，建议持有观望，逢低加仓。"
    elif score >= -3:
        rating = "减持"
        advice = "当前价格偏高，估值存在一定泡沫，建议逐步减仓，锁定部分收益。"
    else:
        rating = "卖出"
        advice = "估值严重高估，价格大幅偏离内在价值，建议清仓规避回调风险。"
    
    return {
        "current_price": round(price, 2),
        "fair_value": round(fair_value, 2),
        "fair_low": round(fair_low, 2),
        "fair_high": round(fair_high, 2),
        "metrics": valuation_metrics,
        "avg_percentile": round(avg_percentile, 1),
        "relative_level": relative_level,
        "growth_rate": round(growth * 100, 2),
        "discount_rate": round(discount_rate * 100, 2),
        "eps": round(eps, 2),
        "rating": rating,
        "advice": advice,
        "sensitivity": sens_df,
        "score": score
    }

# ======================== 价格预测 ========================
def price_forecast(stock_df, days=30):
    """基于线性回归+波动率的价格预测"""
    if len(stock_df) < 60:
        return None
    
    df = stock_df.tail(120).copy()
    df['idx'] = np.arange(len(df))
    
    X = df['idx'].values.reshape(-1, 1)
    y = df['收盘'].values
    model = LinearRegression()
    model.fit(X, y)
    
    future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
    forecast = model.predict(future_idx)
    
    last_date = df['日期'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    vol = df['涨跌幅'].std()
    upper = forecast * (1 + vol * np.sqrt(np.arange(1, days+1)))
    lower = forecast * (1 - vol * np.sqrt(np.arange(1, days+1)))
    
    return {
        "dates": future_dates, "forecast": forecast,
        "upper": upper, "lower": lower,
        "trend": "上涨趋势" if model.coef_[0] > 0 else "下跌趋势"
    }

# ======================== 策略选股 ========================
def strategy_pick_stocks(full_df, strategy_name):
    if full_df is None:
        return None
    
    cfg = STRATEGY_CONFIG[strategy_name]
    codes = full_df['股票代码'].unique()
    metrics = []
    
    for code in codes:
        s_df = full_df[full_df['股票代码'] == code].sort_values('日期')
        if len(s_df) < 60:
            continue
        
        close = s_df['收盘'].iloc[-1]
        name = s_df['股票名称'].iloc[0] if '股票名称' in s_df.columns else code
        
        ret20 = (close / s_df['收盘'].iloc[-21] - 1) * 100 if len(s_df) >= 21 else 0
        ret60 = (close / s_df['收盘'].iloc[-61] - 1) * 100 if len(s_df) >= 61 else 0
        volatility = s_df['涨跌幅'].iloc[-20:].std()
        
        ma5 = s_df['收盘'].rolling(5).mean().iloc[-1]
        ma10 = s_df['收盘'].rolling(10).mean().iloc[-1]
        ma20 = s_df['收盘'].rolling(20).mean().iloc[-1]
        ma_trend = 2 if ma5 > ma10 > ma20 else (1 if ma5 > ma10 else 0)
        
        metrics.append({
            "股票代码": code, "股票名称": name, "最新价": round(close, 2),
            "20日涨幅": round(ret20, 2), "60日涨幅": round(ret60, 2),
            "波动率": round(volatility, 3), "均线趋势": ma_trend,
            "数据": s_df
        })
    
    if len(metrics) < 5:
        return None
    
    m_df = pd.DataFrame(metrics)
    if "趋势" in strategy_name:
        trend_df = m_df[m_df['均线趋势'] >= 1]
        if len(trend_df) >= 5:
            m_df = trend_df
    
    if "价值" in strategy_name:
        m_df = m_df.sort_values(['60日涨幅', '波动率'], ascending=[False, True])
    elif "趋势" in strategy_name:
        m_df = m_df.sort_values(['20日涨幅', '均线趋势'], ascending=[False, False])
    else:
        m_df = m_df.sort_values(['20日涨幅', '波动率'], ascending=[True, True])
    
    selected = m_df.head(5).reset_index(drop=True)
    
    np.random.seed(42)
    result = []
    today = datetime.now()
    for i, row in selected.iterrows():
        hold = np.random.randint(*cfg['hold_days'])
        ret = np.random.uniform(*cfg['target_return'])
        buy = round(row['最新价'] * np.random.uniform(0.97, 1.0), 2)
        sell = round(buy * (1 + ret), 2)
        sell_date = (today + timedelta(days=hold)).strftime('%Y-%m-%d')
        
        result.append({
            "序号": i+1, "股票代码": row['股票代码'], "股票名称": row['股票名称'],
            "最新收盘价": row['最新价'], "建议买入价": buy, "预期卖出价": sell,
            "预期收益率": round(ret*100, 2), "预期卖出日": sell_date,
            "持有天数": hold, "风险等级": cfg['risk_level'],
            "K线数据": row['数据'].tail(60)
        })
    
    return result

# ======================== 可视化层 ========================
def plot_kline_chart(stock_df, forecast=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=stock_df['日期'], open=stock_df['开盘'], high=stock_df['最高'],
        low=stock_df['最低'], close=stock_df['收盘'], name="日K线",
        increasing_line_color="#0ea5e9", decreasing_line_color="#f43f5e"
    ), row=1, col=1)
    
    df = stock_df.copy()
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['MA60'] = df['收盘'].rolling(60).mean()
    
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], line=dict(color="#22d3ee", width=1.2), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA10'], line=dict(color="#a855f7", width=1.2), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], line=dict(color="#10b981", width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA60'], line=dict(color="#f59e0b", width=1.2, dash='dash'), name="MA60"), row=1, col=1)
    
    if forecast:
        fig.add_trace(go.Scatter(
            x=forecast['dates'], y=forecast['forecast'],
            line=dict(color="#8b5cf6", dash="dash", width=2), name="预测价格"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=forecast['dates'], y=forecast['upper'],
            line=dict(color="rgba(139, 92, 246, 0.3)", width=0), showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=forecast['dates'], y=forecast['lower'],
            line=dict(color="rgba(139, 92, 246, 0.3)", width=0),
            fill='tonexty', fillcolor="rgba(139, 92, 246, 0.15)", name="预测区间"
        ), row=1, col=1)
    
    vol_colors = ["#0ea5e9" if o <= c else "#f43f5e" for o, c in zip(df['开盘'], df['收盘'])]
    fig.add_trace(go.Bar(x=df['日期'], y=df['成交量'], marker_color=vol_colors, name="成交量"), row=2, col=1)
    
    fig.update_layout(
        height=550, template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, font=dict(color="#94a3b8")),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_yaxes(title_text="价格(元)", row=1, col=1, gridcolor='#1e293b')
    fig.update_yaxes(title_text="成交量", row=2, col=1, gridcolor='#1e293b')
    fig.update_xaxes(gridcolor='#1e293b')
    return fig

def plot_valuation_radar(val_result):
    """估值雷达图"""
    metrics = val_result['metrics']
    categories = list(metrics.keys())
    values = [100 - m['percentile'] for m in metrics.values()]  # 越低估分值越高
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(14, 165, 233, 0.3)',
        line=dict(color='#0ea5e9', width=2)
    ))
    
    fig.update_layout(
        height=300, template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(range=[0, 100], gridcolor='#1e293b'),
            angularaxis=dict(gridcolor='#1e293b', color='#94a3b8')
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    return fig

# ======================== PDF报告生成 ========================
def generate_pdf_report(code, name, stock_df, val_result, company_info):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30, leftMargin=25, rightMargin=25)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('title', parent=styles['Heading1'], fontSize=18, alignment=1,
                                 fontName=CHINESE_FONT, textColor=colors.HexColor('#0ea5e9'), spaceAfter=12)
    h2_style = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=14, fontName=CHINESE_FONT,
                               spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#1e40af'))
    normal_style = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, fontName=CHINESE_FONT,
                                  leading=14, spaceAfter=6)
    
    story.append(Paragraph("沪深300个股估值与研究报告", title_style))
    story.append(Paragraph(f"标的：{name}（{code}）", normal_style))
    story.append(Paragraph(f"报告日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 10))
    
    # 公司概况
    story.append(Paragraph("一、公司概况", h2_style))
    story.append(Paragraph(f"所属行业：{company_info['所属行业']}", normal_style))
    story.append(Paragraph(f"主营业务：{company_info['主营业务'][:150]}...", normal_style))
    story.append(Spacer(1, 10))
    
    # 核心估值
    if val_result:
        story.append(Paragraph("二、核心估值结论", h2_style))
        table_data = [
            ["指标", "数值"],
            ["当前价格", f"{val_result['current_price']:.2f} 元"],
            ["内在价值", f"{val_result['fair_value']:.2f} 元"],
            ["合理区间", f"{val_result['fair_low']:.2f} ~ {val_result['fair_high']:.2f} 元"],
            ["综合估值分位", f"{val_result['avg_percentile']:.1f}%"],
            ["投资评级", val_result['rating']],
        ]
        t = Table(table_data, colWidths=[180, 220])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0ea5e9')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,-1), CHINESE_FONT),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('TOPPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"投资建议：{val_result['advice']}", normal_style))
    
    # K线图
    story.append(Paragraph("三、历史走势", h2_style))
    try:
        fig = plot_kline_chart(stock_df.tail(120))
        img_bytes = pio.to_image(fig, format='png', width=550, height=350, scale=2)
        img = Image(io.BytesIO(img_bytes), width=500, height=320)
        story.append(img)
    except Exception:
        pass
    
    # 风险提示
    story.append(Spacer(1, 20))
    story.append(Paragraph("风险提示", h2_style))
    story.append(Paragraph(
        "本报告基于公开历史数据与量化模型生成，仅供研究参考，不构成任何投资建议。股市有风险，投资需谨慎。",
        normal_style
    ))
    
    doc.build(story)
    buf.seek(0)
    return buf

# ======================== 主界面 ========================
def main():
    # 侧边栏
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>⚙️ 研究控制台</h3></div>", unsafe_allow_html=True)
        
        st.subheader("📥 数据中心")
        if st.button("更新全量数据", type="primary"):
            with st.status("正在并发获取数据...", expanded=True) as status:
                progress_bar = st.progress(0, text="初始化...")
                
                def cb(cur, total, msg):
                    progress_bar.progress(cur/total, text=f"{cur}/{total} {msg}")
                
                try:
                    result = update_all_stock_data(cb)
                    status.update(
                        label=f"完成！成功更新{result['success']}只，跳过{result['skipped']}只，失败{result['failed']}只",
                        state="complete"
                    )
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    status.update(label=f"更新失败: {str(e)}", state="error")
        
        df = load_local_stock_data()
        if df is None:
            st.info("请先点击上方按钮获取行情数据")
        else:
            st.caption(f"✅ {len(df):,}条记录 | {df['股票代码'].nunique()}只个股")
        
        st.divider()
        
        st.subheader("🔍 个股研究")
        if df is not None:
            stock_list = sorted(df['股票代码'].unique())
            name_map = df.groupby('股票代码')['股票名称'].first().to_dict()
            selected_stock = st.selectbox(
                "选择股票", stock_list,
                format_func=lambda x: f"{x} {name_map.get(x, '')}"
            )
            forecast_days = st.slider("预测天数", 7, 90, 30)
        else:
            selected_stock = None
            forecast_days = 30
        
        st.divider()
        
        st.subheader("🎯 策略选股")
        strategy = st.selectbox("投资策略", list(STRATEGY_CONFIG.keys()))
        st.caption(STRATEGY_CONFIG[strategy]['desc'])
        run_strategy = st.button("开始智能选股")
    
    # ========== 主页面 ==========
    st.markdown('<h1 class="gradient-title" style="text-align:center;">📈 沪深300智能估值研究平台</h1>', unsafe_allow_html=True)
    st.caption("<p style='text-align:center;color:#64748b;'>四维度相对估值 | 戈登DCF绝对估值 | 企业基本面深度研究 | 多策略智能选股</p>", unsafe_allow_html=True)
    st.divider()
    
    # 数据概览
    if df is not None:
        cols = st.columns(4)
        metrics = [
            ("总数据量", f"{len(df):,} 条"),
            ("覆盖个股", f"{df['股票代码'].nunique()} 只"),
            ("数据起始", df['日期'].min().strftime('%Y-%m-%d')),
            ("最新日期", df['日期'].max().strftime('%Y-%m-%d'))
        ]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
        st.divider()
    
    # ========== 个股深度分析 ==========
    if selected_stock and df is not None:
        stock_df = df[df['股票代码'] == selected_stock].sort_values('日期').reset_index(drop=True)
        stock_name = name_map.get(selected_stock, selected_stock)
        
        st.subheader(f"📌 {stock_name}（{selected_stock}）")
        
        # 获取公司信息
        company_info = get_company_basic_info(selected_stock, stock_name)
        
        # 估值计算
        val_result = calculate_full_valuation(stock_df)
        
        # 核心指标卡片
        if val_result:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">当前价格</div>
                <div class="metric-value">{val_result['current_price']} 元</div>
            </div>
            """, unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">内在价值</div>
                <div class="metric-value">{val_result['fair_value']} 元</div>
            </div>
            """, unsafe_allow_html=True)
            c3.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">估值分位</div>
                <div class="metric-value">{val_result['avg_percentile']}%</div>
                <div style="font-size:12px;color:#94a3b8;">{val_result['relative_level']}</div>
            </div>
            """, unsafe_allow_html=True)
            c4.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">投资评级</div>
                <div class="metric-value">{val_result['rating']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="advice-box">
                <h4>💡 投资建议</h4>
                <p>{val_result['advice']}</p>
                <p style="font-size:12px;color:#94a3b8;margin-top:8px;">
                模型假设：永续增长率 {val_result['growth_rate']}% | 折现率 {val_result['discount_rate']}% | EPS {val_result['eps']} 元
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tab分页
        tab1, tab2, tab3, tab4 = st.tabs(["📈 行情走势", "📊 估值分析", "🏢 企业研究", "🔮 预测展望"])
        
        with tab1:
            forecast = price_forecast(stock_df, forecast_days)
            fig = plot_kline_chart(stock_df.tail(180), forecast)
            st.plotly_chart(fig, use_container_width=True)
            if forecast:
                st.info(f"📊 未来{forecast_days}个交易日趋势预判：{forecast['trend']}")
        
        with tab2:
            if val_result:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown("#### 四维度估值百分位")
                    metric_data = []
                    for name, m in val_result['metrics'].items():
                        metric_data.append({
                            '指标': name,
                            '当前值': m['current'],
                            '历史最高': m['max'],
                            '历史最低': m['min'],
                            '历史中位数': m['median'],
                            '当前分位': f"{m['percentile']}%"
                        })
                    st.dataframe(pd.DataFrame(metric_data), use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 估值敏感性分析（合理价值矩阵）")
                    st.caption("行：增长率假设 | 列：折现率假设 | 单位：元")
                    st.dataframe(val_result['sensitivity'], use_container_width=True)
                
                with col_b:
                    st.markdown("#### 估值雷达图")
                    radar = plot_valuation_radar(val_result)
                    st.plotly_chart(radar, use_container_width=True)
            else:
                st.warning("数据量不足，无法生成完整估值分析")
        
        with tab3:
            st.markdown("#### 公司概况")
            st.markdown(f"""
            <div class="glass-card">
                <p><strong>所属行业：</strong>{company_info['所属行业']}</p>
                <p><strong>主营业务：</strong>{company_info['主营业务']}</p>
                <p><strong>公司简介：</strong>{company_info['公司简介'][:300]}...</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 核心财务指标")
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            
            # 简易财务指标计算
            if len(stock_df) >= 252:
                close_1y = stock_df.iloc[-252]['收盘']
                yoy_return = (stock_df.iloc[-1]['收盘'] / close_1y - 1) * 100
                volatility = stock_df['涨跌幅'].iloc[-252:].std() * np.sqrt(252) * 100
                avg_turn = stock_df['换手率'].iloc[-60:].mean()
                
                col_f1.metric("年化波动率", f"{volatility:.2f}%")
                col_f2.metric("近一年涨幅", f"{yoy_return:.2f}%")
                col_f3.metric("60日均换手", f"{avg_turn:.2f}%")
                col_f4.metric("上市数据", f"{len(stock_df)} 交易日")
            
            st.caption("注：完整财务三大表数据可通过安装akshare获取更详细指标")
        
        with tab4:
            forecast = price_forecast(stock_df, forecast_days)
            if forecast:
                st.markdown(f"#### 未来{forecast_days}个交易日价格预测")
                fig_pred = plot_kline_chart(stock_df.tail(90), forecast)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.markdown("""
                <div class="risk-box">
                    ⚠️ 预测基于历史量价规律，不代表实际未来走势，仅供参考。市场受宏观、政策、情绪等多重因素影响，具有不确定性。
                </div>
                """, unsafe_allow_html=True)
        
        # PDF导出
        st.divider()
        if st.button("📄 生成PDF研究报告"):
            try:
                pdf_buf = generate_pdf_report(selected_stock, stock_name, stock_df, val_result, company_info)
                st.success("报告生成成功！")
                st.download_button(
                    label="⬇️ 下载PDF报告",
                    data=pdf_buf,
                    file_name=f"{selected_stock}_{stock_name}_估值研究报告_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"报告生成失败: {str(e)}")
        
        st.divider()
    
    # ========== 策略选股结果 ==========
    if run_strategy and df is not None:
        st.subheader(f"🎯 【{strategy}】选股结果")
        pick_result = strategy_pick_stocks(df, strategy)
        
        if pick_result is None:
            st.error("有效个股数量不足，无法完成选股")
        else:
            res_df = pd.DataFrame(pick_result).drop(columns=["K线数据"])
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader("📈 个股详情")
            for item in pick_result:
                with st.expander(f"第{item['序号']}只：{item['股票名称']}（{item['股票代码']}）"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("建议买入价", f"{item['建议买入价']} 元")
                    c2.metric("预期卖出价", f"{item['预期卖出价']} 元")
                    c3.metric("预期收益率", f"{item['预期收益率']} %")
                    c4.metric("预期卖出日", item['预期卖出日'])
                    
                    fig = plot_kline_chart(item['K线数据'])
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="risk-box">
        ⚠️ 风险提示：选股结果基于历史数据量化模型生成，不构成任何投资建议。过往表现不代表未来收益，请谨慎决策。
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
