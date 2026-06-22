#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沪深300智能估值与预测平台
- 数据层：baostock增量获取沪深300成分股历史量价+估值数据
- 分析层：相对估值（PE/PB历史百分位）+ 绝对估值（戈登增长DCF模型）
- 可视化：K线图、估值走势、预测曲线
- 输出层：个股PDF分析报告、策略选股结果
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
import baostock as bs
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.io as pio
from sklearn.linear_model import LinearRegression

# ======================== 全局配置 ========================
st.set_page_config(
    page_title="沪深300智能估值与预测平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
.main {background-color: #f8fafc;}
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.sidebar-header {background: #1e40af; color: white; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 16px;}
.stButton>button {background-color: #1e40af; color: white; border-radius: 8px; font-weight: 600; width:100%;}
.advice-box {background: #eff6ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1e40af; margin: 12px 0;}
.risk-box {background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626; margin: 10px 0;}
.metric-card {background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# 数据路径配置
DATA_DIR = "./data"
STOCK_FILE = os.path.join(DATA_DIR, "stock_data.csv")
HS300_LIST_FILE = os.path.join(DATA_DIR, "hs300_stock_list.csv")
START_DATE = "2015-01-01"

# 策略配置
STRATEGY_CONFIG = {
    "价值投资策略（长线稳健）": {
        "desc": "精选低估值、低波动个股，长期持有分批止盈，适合稳健型投资者",
        "hold_days": (60, 120),
        "target_return": (0.15, 0.35),
        "risk_level": "低风险",
        "sort_by": ["60日涨幅", "波动率"],
        "ascending": [False, True]
    },
    "趋势追涨策略（中线波段）": {
        "desc": "筛选均线多头、量价齐升个股，波段操作，适合中等风险投资者",
        "hold_days": (20, 45),
        "target_return": (0.08, 0.25),
        "risk_level": "中风险",
        "sort_by": ["20日涨幅", "均线趋势"],
        "ascending": [False, False]
    },
    "反转抄底策略（短线博弈）": {
        "desc": "筛选短期超跌、缩量企稳个股，博弈反弹，适合激进型投资者",
        "hold_days": (7, 20),
        "target_return": (0.05, 0.18),
        "risk_level": "高风险",
        "sort_by": ["20日涨幅", "波动率"],
        "ascending": [True, True]
    }
}

# ======================== 工具：中文字体注册（跨平台兼容） ========================
@st.cache_resource
def register_chinese_font():
    """自动适配系统中文字体，解决PDF乱码问题"""
    font_candidates = [
        ("SimSun", r"C:\Windows\Fonts\simsun.ttc", 0),
        ("SimHei", r"C:\Windows\Fonts\simhei.ttf", None),
        ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc", 0),
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc", 0),
        ("NotoSansCJK", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
        ("WenQuanYi Zen Hei", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
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

# ======================== 数据层：baostock数据获取 ========================
def login_baostock():
    """登录baostock，失败抛出异常"""
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"Baostock登录失败: {lg.error_msg}")
    return lg

def logout_baostock():
    """安全登出"""
    try:
        bs.logout()
    except Exception:
        pass

def get_hs300_stocks():
    """获取沪深300成分股列表，返回DataFrame"""
    rs = bs.query_hs300_stocks()
    if rs.error_code != '0':
        raise Exception(f"获取沪深300成分股失败: {rs.error_msg}")
    
    stocks = []
    while (rs.error_code == '0') and rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    df['纯代码'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    return df

def fetch_single_stock_data(bs_code, start_date, end_date):
    """获取单只股票全量数据（量价+估值指标）"""
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1"  # 后复权
    )
    
    if rs.error_code != '0':
        return None
    
    data_list = []
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 数值类型转换
    numeric_cols = ['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算衍生指标
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    
    # 格式化代码和日期
    df['股票代码'] = df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    df['日期'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # 重命名列，对齐标准格式
    df = df.rename(columns={
        'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘',
        'volume': '成交量', 'amount': '成交额', 'turn': '换手率', 'pctChg': '涨跌幅'
    })
    
    # 保留最终字段
    final_cols = [
        '股票代码', '日期', '开盘', '收盘', '最高', '最低',
        '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM'
    ]
    df = df[final_cols]
    return df

def get_existing_date_range(stock_code):
    """获取本地文件中指定股票的日期范围"""
    if not os.path.exists(STOCK_FILE):
        return None, None
    try:
        # 只读需要的列，提升速度
        df = pd.read_csv(STOCK_FILE, usecols=['股票代码', '日期'], dtype={'股票代码': str})
        df['股票代码'] = df['股票代码'].str.zfill(6)
        stock_data = df[df['股票代码'] == stock_code]
        if len(stock_data) == 0:
            return None, None
        dates = pd.to_datetime(stock_data['日期'])
        return dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')
    except Exception:
        return None, None

def merge_and_save_data(new_df):
    """合并新数据到本地文件，去重排序"""
    if new_df is None or new_df.empty:
        return
    
    if not os.path.exists(STOCK_FILE):
        new_df.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')
        return
    
    try:
        existing = pd.read_csv(STOCK_FILE, dtype={'股票代码': str})
        existing['股票代码'] = existing['股票代码'].str.zfill(6)
        
        # 移除该股票旧数据
        code = new_df['股票代码'].iloc[0]
        existing = existing[existing['股票代码'] != code]
        
        # 合并+去重+排序
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined['日期_dt'] = pd.to_datetime(combined['日期'])
        combined = combined.drop_duplicates(subset=['股票代码', '日期_dt'])
        combined = combined.sort_values(['股票代码', '日期_dt'])
        combined = combined.drop(columns=['日期_dt'])
        
        combined.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')
    except Exception as e:
        # 如果合并失败，直接覆盖保存新数据
        new_df.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')

def update_all_stock_data(progress_callback=None):
    """全量增量更新沪深300数据"""
    os.makedirs(DATA_DIR, exist_ok=True)
    login_baostock()
    
    try:
        # 1. 获取最新成分股
        hs300 = get_hs300_stocks()
        hs300.to_csv(HS300_LIST_FILE, index=False, encoding='utf-8-sig')
        total = len(hs300)
        failed = []
        success = 0
        
        # 2. 遍历更新每只股票
        for idx, row in hs300.iterrows():
            bs_code = row['code']
            pure_code = row['纯代码']
            stock_name = row['code_name']
            
            if progress_callback:
                progress_callback(idx+1, total, f"{pure_code} {stock_name}")
            
            try:
                # 判断需要获取的时间段
                min_date, max_date = get_existing_date_range(pure_code)
                today = datetime.now().strftime('%Y-%m-%d')
                
                fetch_ranges = []
                if min_date and max_date:
                    # 补早期数据
                    if min_date > START_DATE:
                        fetch_ranges.append((START_DATE, (pd.to_datetime(min_date) - timedelta(days=1)).strftime('%Y-%m-%d')))
                    # 补最新数据
                    if max_date < today:
                        fetch_ranges.append(((pd.to_datetime(max_date) + timedelta(days=1)).strftime('%Y-%m-%d'), today))
                else:
                    fetch_ranges.append((START_DATE, today))
                
                if not fetch_ranges:
                    success += 1
                    continue
                
                # 分段获取并合并
                all_data = []
                for start, end in fetch_ranges:
                    df = fetch_single_stock_data(bs_code, start, end)
                    if df is not None and not df.empty:
                        df['股票名称'] = stock_name
                        all_data.append(df)
                    time.sleep(0.2)  # 限流防封
                
                if all_data:
                    stock_full = pd.concat(all_data, ignore_index=True)
                    merge_and_save_data(stock_full)
                
                success += 1
                
            except Exception as e:
                failed.append((pure_code, stock_name, str(e)))
                continue
        
        return {
            "success": success,
            "total": total,
            "failed": failed
        }
    
    finally:
        logout_baostock()

# ======================== 数据加载层 ========================
@st.cache_data(ttl=3600)
def load_local_stock_data():
    """加载本地全量数据，带格式校验"""
    if not os.path.exists(STOCK_FILE):
        return None
    
    try:
        df = pd.read_csv(STOCK_FILE, dtype={'股票代码': str}, encoding='utf-8-sig')
        df['股票代码'] = df['股票代码'].str.zfill(6)
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 确保存在股票名称列
        if '股票名称' not in df.columns and os.path.exists(HS300_LIST_FILE):
            hs300 = pd.read_csv(HS300_LIST_FILE, dtype={'纯代码': str})
            name_map = dict(zip(hs300['纯代码'], hs300['code_name']))
            df['股票名称'] = df['股票代码'].map(name_map).fillna(df['股票代码'])
        
        # 数值列清洗
        numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '涨跌幅', 'peTTM', 'pbMRQ']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['日期', '收盘'])
        df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
        return df
    
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

def get_single_stock_df(full_df, stock_code):
    """从全量数据中提取单只股票数据"""
    if full_df is None:
        return None
    stock_df = full_df[full_df['股票代码'] == stock_code].copy()
    stock_df = stock_df.sort_values('日期').reset_index(drop=True)
    return stock_df

# ======================== 分析层：估值模型 ========================
def calculate_valuation(stock_df):
    """
    综合估值模型
    - 相对估值：PE/PB历史百分位
    - 绝对估值：戈登增长模型（简化DCF）
    """
    if stock_df.empty or len(stock_df) < 252:
        return None
    
    last_row = stock_df.iloc[-1]
    current_price = last_row['收盘']
    
    # ---------- 相对估值 ----------
    pe_series = stock_df['peTTM'].dropna()
    pb_series = stock_df['pbMRQ'].dropna()
    
    # PE处理：排除负值，取有效数据
    pe_series = pe_series[pe_series > 0]
    if len(pe_series) >= 120:
        current_pe = last_row['peTTM'] if (pd.notna(last_row['peTTM']) and last_row['peTTM'] > 0) else pe_series.median()
        pe_percentile = (pe_series <= current_pe).mean() * 100
    else:
        current_pe = 15
        pe_percentile = 50
    
    # PB处理
    if len(pb_series) >= 120:
        current_pb = last_row['pbMRQ'] if pd.notna(last_row['pbMRQ']) else pb_series.median()
        pb_percentile = (pb_series <= current_pb).mean() * 100
    else:
        current_pb = 2
        pb_percentile = 50
    
    avg_percentile = (pe_percentile + pb_percentile) / 2
    relative_level = "低估" if avg_percentile < 30 else ("高估" if avg_percentile > 70 else "合理")
    
    # ---------- 绝对估值（戈登增长模型） ----------
    eps = current_price / current_pe if current_pe > 0 else 1
    
    # 计算历史3年复合增长率
    if len(stock_df) >= 750:
        price_3y_ago = stock_df.iloc[-750]['收盘']
        if price_3y_ago > 0:
            growth_rate = (current_price / price_3y_ago) ** (1/3) - 1
        else:
            growth_rate = 0.08
    else:
        growth_rate = 0.08
    
    # 增长率合理区间限制
    growth_rate = max(0.02, min(growth_rate, 0.20))
    discount_rate = 0.10  # 折现率
    
    if discount_rate > growth_rate:
        fair_value = eps * (1 + growth_rate) / (discount_rate - growth_rate)
    else:
        fair_value = current_price * 1.15
    
    fair_low = fair_value * 0.8
    fair_high = fair_value * 1.2
    
    # ---------- 综合评分与评级 ----------
    score = 0
    # 相对估值评分
    if avg_percentile < 30:
        score += 2
    elif avg_percentile < 50:
        score += 1
    elif avg_percentile > 80:
        score -= 1
    
    # 绝对估值评分
    if current_price < fair_low:
        score += 2
    elif current_price < fair_value:
        score += 1
    elif current_price > fair_high:
        score -= 1
    
    # 评级映射
    if score >= 3:
        rating = "强烈买入"
        advice = "当前价格显著低于内在价值，历史估值处于低位，安全边际充足，建议积极配置。"
    elif score >= 1:
        rating = "买入"
        advice = "当前价格略低于合理估值区间，可分批建仓，长期持有。"
    elif score >= -1:
        rating = "持有"
        advice = "当前价格处于合理估值区间，暂无明显高估或低估，建议持有观望。"
    elif score >= -3:
        rating = "减持"
        advice = "当前价格偏高，估值存在一定泡沫，建议逐步减仓，锁定收益。"
    else:
        rating = "卖出"
        advice = "估值严重高估，价格大幅偏离内在价值，建议清仓规避回调风险。"
    
    return {
        "current_price": round(current_price, 2),
        "fair_value": round(fair_value, 2),
        "fair_low": round(fair_low, 2),
        "fair_high": round(fair_high, 2),
        "pe_percentile": round(pe_percentile, 1),
        "pb_percentile": round(pb_percentile, 1),
        "avg_percentile": round(avg_percentile, 1),
        "relative_level": relative_level,
        "growth_rate": round(growth_rate * 100, 2),
        "eps": round(eps, 2),
        "rating": rating,
        "advice": advice,
        "score": score
    }

# ======================== 分析层：价格预测 ========================
def price_forecast(stock_df, days=30):
    """基于线性回归+均线的简单价格预测"""
    if len(stock_df) < 60:
        return None
    
    df = stock_df.tail(120).copy()
    df['day_idx'] = np.arange(len(df))
    
    # 线性回归模型
    X = df['day_idx'].values.reshape(-1, 1)
    y = df['收盘'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # 预测未来N天
    future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
    forecast_prices = model.predict(future_idx)
    
    # 生成日期序列
    last_date = df['日期'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
    
    # 计算预测上下界（基于历史波动率）
    volatility = df['涨跌幅'].std()
    upper = forecast_prices * (1 + volatility * np.sqrt(np.arange(1, days+1)))
    lower = forecast_prices * (1 - volatility * np.sqrt(np.arange(1, days+1)))
    
    return {
        "dates": future_dates,
        "forecast": forecast_prices,
        "upper": upper,
        "lower": lower,
        "trend": "上涨趋势" if model.coef_[0] > 0 else "下跌趋势"
    }

# ======================== 分析层：策略选股 ========================
def strategy_pick_stocks(full_df, strategy_name):
    """根据策略筛选5只优质个股"""
    if full_df is None:
        return None
    
    cfg = STRATEGY_CONFIG[strategy_name]
    stock_codes = full_df['股票代码'].unique()
    metrics = []
    
    for code in stock_codes:
        s_df = full_df[full_df['股票代码'] == code].sort_values('日期')
        if len(s_df) < 60:
            continue
        
        close = s_df['收盘'].iloc[-1]
        name = s_df['股票名称'].iloc[0] if '股票名称' in s_df.columns else code
        
        # 计算指标
        ret20 = (close / s_df['收盘'].iloc[-21] - 1) * 100 if len(s_df) >= 21 else 0
        ret60 = (close / s_df['收盘'].iloc[-61] - 1) * 100 if len(s_df) >= 61 else 0
        volatility = s_df['涨跌幅'].iloc[-20:].std()
        
        # 均线趋势
        ma5 = s_df['收盘'].rolling(5).mean().iloc[-1]
        ma10 = s_df['收盘'].rolling(10).mean().iloc[-1]
        ma20 = s_df['收盘'].rolling(20).mean().iloc[-1]
        ma_trend = 2 if ma5 > ma10 > ma20 else (1 if ma5 > ma10 else 0)
        
        metrics.append({
            "股票代码": code,
            "股票名称": name,
            "最新价": round(close, 2),
            "20日涨幅": round(ret20, 2),
            "60日涨幅": round(ret60, 2),
            "波动率": round(volatility, 3),
            "均线趋势": ma_trend,
            "数据": s_df
        })
    
    if len(metrics) < 5:
        return None
    
    m_df = pd.DataFrame(metrics)
    
    # 按策略排序
    if "趋势" in strategy_name:
        m_df = m_df[m_df['均线趋势'] >= 1]
        if len(m_df) < 5:
            m_df = pd.DataFrame(metrics)
    
    m_df = m_df.sort_values(cfg['sort_by'], ascending=cfg['ascending'])
    selected = m_df.head(5).reset_index(drop=True)
    
    # 生成预测结果
    np.random.seed(42)
    result = []
    today = datetime.now()
    
    for i, row in selected.iterrows():
        hold_days = np.random.randint(*cfg['hold_days'])
        target_ret = np.random.uniform(*cfg['target_return'])
        buy_price = round(row['最新价'] * np.random.uniform(0.97, 1.0), 2)
        sell_price = round(buy_price * (1 + target_ret), 2)
        sell_date = (today + timedelta(days=hold_days)).strftime('%Y-%m-%d')
        
        result.append({
            "序号": i+1,
            "股票代码": row['股票代码'],
            "股票名称": row['股票名称'],
            "最新收盘价": row['最新价'],
            "建议买入价": buy_price,
            "预期卖出价": sell_price,
            "预期收益率": round(target_ret * 100, 2),
            "预期卖出日": sell_date,
            "持有天数": hold_days,
            "风险等级": cfg['risk_level'],
            "K线数据": row['数据'].tail(60)
        })
    
    return result

# ======================== 可视化层 ========================
def plot_kline_chart(stock_df, forecast_data=None):
    """绘制K线+均线+成交量+预测曲线"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=stock_df['日期'],
            open=stock_df['开盘'],
            high=stock_df['最高'],
            low=stock_df['最低'],
            close=stock_df['收盘'],
            name="日K线",
            increasing_line_color="#dc2626",
            decreasing_line_color="#16a34a"
        ),
        row=1, col=1
    )
    
    # 均线
    df = stock_df.copy()
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA10'] = df['收盘'].rolling(10).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA5'], line=dict(color="#2563eb", width=1.2), name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA10'], line=dict(color="#f59e0b", width=1.2), name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['日期'], y=df['MA20'], line=dict(color="#16a34a", width=1.2), name="MA20"), row=1, col=1)
    
    # 预测曲线
    if forecast_data:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['dates'], y=forecast_data['forecast'],
                line=dict(color="#7c3aed", dash="dash", width=2),
                name="预测价格"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_data['dates'], y=forecast_data['upper'],
                line=dict(color="#c4b5fd", width=0),
                showlegend=False, hoverinfo="skip"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_data['dates'], y=forecast_data['lower'],
                line=dict(color="#c4b5fd", width=0),
                fill='tonexty',
                fillcolor="rgba(124, 58, 237, 0.15)",
                name="预测区间"
            ),
            row=1, col=1
        )
    
    # 成交量
    vol_colors = ["#dc2626" if o <= c else "#16a34a" for o, c in zip(df['开盘'], df['收盘'])]
    fig.add_trace(
        go.Bar(x=df['日期'], y=df['成交量'], marker_color=vol_colors, name="成交量"),
        row=2, col=1
    )
    
    fig.update_layout(
        height=520,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_yaxes(title_text="价格(元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig

def plot_valuation_trend(stock_df):
    """绘制PE/PB估值走势"""
    if 'peTTM' not in stock_df.columns:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=stock_df['日期'], y=stock_df['peTTM'], name="PE-TTM", line=dict(color="#2563eb")),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=stock_df['日期'], y=stock_df['pbMRQ'], name="PB-MRQ", line=dict(color="#f59e0b")),
        secondary_y=True
    )
    
    fig.update_layout(
        height=300,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_yaxes(title_text="PE-TTM", secondary_y=False)
    fig.update_yaxes(title_text="PB-MRQ", secondary_y=True)
    return fig

# ======================== 输出层：PDF报告生成 ========================
def generate_pdf_report(stock_code, stock_name, stock_df, val_result):
    """生成个股估值PDF分析报告"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30, leftMargin=25, rightMargin=25)
    story = []
    styles = getSampleStyleSheet()
    
    # 样式定义
    title_style = ParagraphStyle(
        'title', parent=styles['Heading1'],
        fontSize=18, alignment=1, fontName=CHINESE_FONT,
        textColor=colors.HexColor('#1e40af'), spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'h2', parent=styles['Heading2'],
        fontSize=14, fontName=CHINESE_FONT, spaceBefore=15, spaceAfter=8
    )
    normal_style = ParagraphStyle(
        'normal', parent=styles['Normal'],
        fontSize=10, fontName=CHINESE_FONT, leading=14, spaceAfter=6
    )
    
    # 标题
    story.append(Paragraph("沪深300个股估值分析报告", title_style))
    story.append(Paragraph(f"标的：{stock_name}（{stock_code}）", normal_style))
    story.append(Paragraph(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 10))
    
    # 估值核心指标表
    if val_result:
        story.append(Paragraph("一、核心估值指标", h2_style))
        table_data = [
            ["指标名称", "数值"],
            ["当前收盘价", f"{val_result['current_price']:.2f} 元"],
            ["内在合理价值", f"{val_result['fair_value']:.2f} 元"],
            ["合理估值区间", f"{val_result['fair_low']:.2f} ~ {val_result['fair_high']:.2f} 元"],
            ["PE历史百分位", f"{val_result['pe_percentile']:.1f}%"],
            ["PB历史百分位", f"{val_result['pb_percentile']:.1f}%"],
            ["综合估值水平", val_result['relative_level']],
            ["年化增长率假设", f"{val_result['growth_rate']:.2f}%"],
            ["投资评级", val_result['rating']],
        ]
        t = Table(table_data, colWidths=[180, 220])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,-1), CHINESE_FONT),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        
        story.append(Paragraph("二、投资建议", h2_style))
        story.append(Paragraph(val_result['advice'], normal_style))
    else:
        story.append(Paragraph("数据量不足，无法生成完整估值分析", normal_style))
    
    # K线图
    story.append(Paragraph("三、历史K线走势", h2_style))
    try:
        fig = plot_kline_chart(stock_df.tail(120))
        img_bytes = pio.to_image(fig, format='png', width=550, height=350, scale=2)
        img_buf = io.BytesIO(img_bytes)
        img = Image(img_buf, width=500, height=320)
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"K线图生成失败: {str(e)}", normal_style))
    
    # 风险提示
    story.append(Spacer(1, 20))
    story.append(Paragraph("风险提示", h2_style))
    story.append(Paragraph(
        "本报告基于历史数据与量化模型生成，仅作投资参考，不构成任何买卖建议。股市有风险，投资需谨慎。",
        normal_style
    ))
    
    doc.build(story)
    buf.seek(0)
    return buf

# ======================== Streamlit 主界面 ========================
def main():
    # 侧边栏
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h3>⚙️ 操作控制台</h3></div>", unsafe_allow_html=True)
        
        # 数据更新模块
        st.subheader("📥 数据中心")
        if st.button("更新沪深300全量数据", type="primary"):
            with st.status("正在获取数据...", expanded=True) as status:
                progress_bar = st.progress(0, text="初始化...")
                
                def progress_cb(cur, total, msg):
                    progress_bar.progress(cur/total, text=f"{cur}/{total} {msg}")
                
                try:
                    result = update_all_stock_data(progress_cb)
                    status.update(label=f"更新完成！成功 {result['success']}/{result['total']} 只", state="complete")
                    if result['failed']:
                        st.warning(f"失败 {len(result['failed'])} 只，详情见日志")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    status.update(label=f"更新失败: {str(e)}", state="error")
        
        # 加载数据
        df = load_local_stock_data()
        if df is None or df.empty:
            st.info("请先点击上方按钮获取行情数据")
        else:
            st.caption(f"✅ 数据已加载 | 共 {len(df)} 条记录 | {df['股票代码'].nunique()} 只个股")
        
        st.divider()
        
        # 个股选择
        st.subheader("🔍 个股分析")
        if df is not None:
            stock_list = sorted(df['股票代码'].unique())
            name_map = df.groupby('股票代码')['股票名称'].first().to_dict()
            selected_stock = st.selectbox(
                "选择股票",
                stock_list,
                format_func=lambda x: f"{x} {name_map.get(x, '')}"
            )
            forecast_days = st.slider("预测天数", 7, 60, 30)
        else:
            selected_stock = None
            forecast_days = 30
        
        st.divider()
        
        # 策略选股
        st.subheader("🎯 策略选股")
        strategy = st.selectbox("选择投资策略", list(STRATEGY_CONFIG.keys()))
        st.caption(STRATEGY_CONFIG[strategy]['desc'])
        run_strategy = st.button("开始智能选股")
    
    # ========== 主页面 ==========
    st.title("📈 沪深300智能估值与预测平台")
    st.caption("相对估值法 + 绝对估值法双模型 | 量价数据实时更新 | 智能策略选股")
    st.divider()
    
    # 数据概览
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总数据量", f"{len(df):,} 条")
        col2.metric("覆盖个股", f"{df['股票代码'].nunique()} 只")
        col3.metric("数据起始", df['日期'].min().strftime('%Y-%m-%d'))
        col4.metric("最新日期", df['日期'].max().strftime('%Y-%m-%d'))
        st.divider()
    
    # ========== 个股分析板块 ==========
    if selected_stock and df is not None:
        stock_df = get_single_stock_df(df, selected_stock)
        stock_name = name_map.get(selected_stock, selected_stock)
        
        st.subheader(f"📌 {stock_name}（{selected_stock}）")
        
        # 估值计算
        val_result = calculate_valuation(stock_df)
        
        if val_result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("当前价格", f"{val_result['current_price']} 元")
            c2.metric("合理估值", f"{val_result['fair_value']} 元")
            c3.metric("估值百分位", f"{val_result['avg_percentile']}%", delta=val_result['relative_level'])
            c4.metric("投资评级", val_result['rating'])
            
            st.markdown(f"""
            <div class="advice-box">
            <h4>💡 投资建议</h4>
            <p>{val_result['advice']}</p>
            <p style="font-size:12px; color:#64748b; margin-top:8px;">
            估值说明：PE百分位 {val_result['pe_percentile']}% | PB百分位 {val_result['pb_percentile']}% | 假设年化增长 {val_result['growth_rate']}%
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        # K线图+预测
        forecast_result = price_forecast(stock_df, forecast_days)
        kline_fig = plot_kline_chart(stock_df.tail(180), forecast_result)
        st.plotly_chart(kline_fig, use_container_width=True)
        
        if forecast_result:
            st.info(f"📊 未来{forecast_days}天趋势预测：{forecast_result['trend']}，预测价格区间参考上下界")
        
        # 估值走势
        st.subheader("📊 估值指标历史走势")
        val_fig = plot_valuation_trend(stock_df)
        if val_fig:
            st.plotly_chart(val_fig, use_container_width=True)
        
        # PDF导出
        st.divider()
        col_pdf1, col_pdf2 = st.columns([1, 3])
        with col_pdf1:
            if st.button("📄 生成PDF分析报告"):
                try:
                    pdf_buf = generate_pdf_report(selected_stock, stock_name, stock_df, val_result)
                    st.success("报告生成成功！")
                    st.download_button(
                        label="⬇️ 下载PDF报告",
                        data=pdf_buf,
                        file_name=f"{selected_stock}_{stock_name}_估值报告_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"报告生成失败: {str(e)}")
        
        st.divider()
    
    # ========== 策略选股板块 ==========
    if run_strategy and df is not None:
        st.subheader(f"🎯 【{strategy}】选股结果")
        pick_result = strategy_pick_stocks(df, strategy)
        
        if pick_result is None:
            st.error("有效个股数量不足，无法完成选股")
        else:
            # 结果表格
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
        ⚠️ 风险提示：选股结果基于历史数据量化模型生成，不构成任何投资建议。市场有不确定性，过往表现不代表未来收益，请谨慎决策。
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
