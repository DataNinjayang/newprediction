#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time
import io
import re
import baostock as bs
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.io as pio

# ======================== 页面配置 ========================
st.set_page_config(page_title="沪深300智能估值平台", page_icon="📊", layout="wide")

# ======================== 自定义样式 ========================
st.markdown("""
<style>
.main {background-color: #f8fafc;}
.block-container {padding-top: 1rem;}
.sidebar-header {background: #1e40af; color: white; padding: 10px; border-radius: 8px; text-align: center;}
.stButton>button {background-color: #1e40af; color: white; border-radius: 8px; font-weight: 600; width:100%;}
.advice-box {background: #eff6ff; padding: 20px; border-radius: 10px; border-left: 4px solid #1e40af;}
.risk-box {background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;}
</style>
""", unsafe_allow_html=True)

# ======================== 数据文件路径 ========================
DATA_DIR = "./data"
STOCK_FILE = os.path.join(DATA_DIR, "stock_data.csv")
START_DATE = "2015-01-01"

# ======================== 数据获取函数 ========================
def login_baostock():
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"登录失败: {lg.error_msg}")
    return lg

def logout_baostock():
    bs.logout()

def fetch_stock_history(bs_code, start, end):
    """获取单只股票历史数据（含估值指标）"""
    rs = bs.query_history_k_data_plus(bs_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date=start, end_date=end,
        frequency="d", adjustflag="1")
    if rs.error_code != '0':
        return None
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    df = pd.DataFrame(data, columns=rs.fields)
    numeric_cols = ['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['振幅'] = ((df['high'] - df['low']) / df['preclose'] * 100).round(2)
    df['涨跌额'] = (df['close'] - df['preclose']).round(2)
    df['日期'] = pd.to_datetime(df['date']).dt.strftime('%Y/%m/%d')
    df = df[['日期','open','high','low','close','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM','振幅','涨跌额']]
    df.columns = ['日期','开盘','最高','最低','收盘','成交量','成交额','换手率','涨跌幅','peTTM','pbMRQ','psTTM','pcfNcfTTM','振幅','涨跌额']
    return df

def get_all_data(progress_callback=None):
    """获取全部沪深300成分股数据（增量更新）"""
    os.makedirs(DATA_DIR, exist_ok=True)
    login_baostock()
    try:
        # 获取成分股列表
        rs = bs.query_hs300_stocks()
        stocks = []
        while (rs.error_code == '0') & rs.next():
            stocks.append(rs.get_row_data())
        hs300_df = pd.DataFrame(stocks, columns=rs.fields)
        hs300_df['纯代码'] = hs300_df['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
        code_name = dict(zip(hs300_df['纯代码'], hs300_df['code_name']))
        total = len(hs300_df)

        # 加载已有数据
        existing_df = pd.DataFrame()
        if os.path.exists(STOCK_FILE):
            existing_df = pd.read_csv(STOCK_FILE, encoding='utf-8-sig')
            existing_df['日期'] = pd.to_datetime(existing_df['日期'])
            existing_df['股票代码'] = existing_df['股票代码'].astype(str).str.zfill(6)

        today = datetime.now().date()
        for idx, row in hs300_df.iterrows():
            code = row['纯代码']
            bs_code = row['code']
            name = row['code_name']
            if progress_callback:
                progress_callback(idx+1, total, f"{code} {name}")

            # 检查已有数据
            existing_stock = existing_df[existing_df['股票代码'] == code] if not existing_df.empty else pd.DataFrame()
            if not existing_stock.empty:
                existing_dates = existing_stock['日期'].dt.date
                min_date = existing_dates.min()
                max_date = existing_dates.max()
                if min_date <= datetime.strptime(START_DATE, '%Y-%m-%d').date() and max_date >= today - timedelta(days=1):
                    continue  # 数据完整
                # 补缺
                if min_date > datetime.strptime(START_DATE, '%Y-%m-%d').date():
                    fetch_start = START_DATE
                else:
                    fetch_start = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
                if max_date < today:
                    fetch_end = today.strftime('%Y-%m-%d')
                else:
                    continue
            else:
                fetch_start = START_DATE
                fetch_end = today.strftime('%Y-%m-%d')

            new_data = fetch_stock_history(bs_code, fetch_start, fetch_end)
            if new_data is not None and not new_data.empty:
                new_data['股票代码'] = code
                new_data['股票名称'] = name
                if not existing_stock.empty:
                    combined = pd.concat([existing_stock, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['日期']).sort_values('日期')
                    existing_df = existing_df[existing_df['股票代码'] != code]
                    existing_df = pd.concat([existing_df, combined], ignore_index=True)
                else:
                    existing_df = pd.concat([existing_df, new_data], ignore_index=True)
                # 每只股票保存一次（防中断丢失）
                existing_df.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')
            time.sleep(0.3)  # 避免请求过频

        existing_df.to_csv(STOCK_FILE, index=False, encoding='utf-8-sig')
        return True
    finally:
        logout_baostock()

def load_local_data():
    if os.path.exists(STOCK_FILE):
        df = pd.read_csv(STOCK_FILE, encoding='utf-8-sig')
        df['日期'] = pd.to_datetime(df['日期'])
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
        return df
    return None

# ======================== 估值模型 ========================
def calculate_valuation(stock_df):
    """返回估值结果字典"""
    if stock_df.empty or len(stock_df) < 252:
        return None
    last = stock_df.iloc[-1]
    price = last['收盘']

    # 相对估值（历史百分位）
    pe_series = stock_df['peTTM'].dropna()
    pb_series = stock_df['pbMRQ'].dropna()
    if len(pe_series) < 100 or len(pb_series) < 100:
        pe_ratio = pe_series.mean() if not pe_series.empty else 15
        pb_ratio = pb_series.mean() if not pb_series.empty else 2
        pe_percentile = 50
        pb_percentile = 50
    else:
        pe_ratio = last['peTTM'] if not np.isnan(last['peTTM']) else pe_series.mean()
        pb_ratio = last['pbMRQ'] if not np.isnan(last['pbMRQ']) else pb_series.mean()
        pe_percentile = (pe_series <= pe_ratio).mean() * 100
        pb_percentile = (pb_series <= pb_ratio).mean() * 100

    # 绝对估值（简化DCF：戈登增长模型）
    eps = price / pe_ratio if pe_ratio > 0 else 1
    # 增长率取过去3年价格复合增长
    if len(stock_df) >= 756:
        price_3y = stock_df.iloc[-756]['收盘']
        growth = (price / price_3y) ** (1/3) - 1
    else:
        growth = 0.08
    growth = min(max(growth, 0.02), 0.20)
    r = 0.10  # 折现率
    if r > growth:
        fair_value = eps * (1 + growth) / (r - growth)
    else:
        fair_value = price * 1.2
    fair_low = fair_value * 0.8
    fair_high = fair_value * 1.2

    # 综合评分
    score = 0
    if pe_percentile < 30: score += 2
    elif pe_percentile < 50: score += 1
    elif pe_percentile > 80: score -= 1

    if price < fair_low: score += 2
    elif price < fair_value: score += 1
    elif price > fair_high: score -= 1

    if score >= 3:
        rating = "强烈买入"
        advice = "当前价格显著低于内在价值，且历史估值处于低位，建议积极配置。"
    elif score >= 1:
        rating = "买入"
        advice = "当前价格略低于合理估值，可分批建仓。"
    elif score >= -1:
        rating = "持有"
        advice = "当前价格处于合理区间，建议持有观望。"
    elif score >= -3:
        rating = "减持"
        advice = "当前价格偏高，建议逐步减仓。"
    else:
        rating = "卖出"
        advice = "估值严重高估，建议清仓规避风险。"

    relative = "低估" if pe_percentile < 30 else ("高估" if pe_percentile > 70 else "合理")

    return {
        'current_price': price,
        'fair_value': fair_value,
        'fair_low': fair_low,
        'fair_high': fair_high,
        'percentile': (pe_percentile + pb_percentile) / 2,
        'relative': relative,
        'rating': rating,
        'advice': advice,
        'eps': eps,
        'growth': growth
    }

# ======================== PDF生成 ========================
def register_chinese_font():
    font_name = "SimSun"
    try:
        font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simsun.ttc")
        pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
    except:
        try:
            font_path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simhei.ttf")
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except:
            pass
    return font_name

CHINESE_FONT = register_chinese_font()

def create_pdf_report(code, name, stock_df, val_result):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=1, fontName=CHINESE_FONT)
    head_style = ParagraphStyle('Head', parent=styles['Heading2'], fontSize=14, fontName=CHINESE_FONT)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=10, fontName=CHINESE_FONT)

    story.append(Paragraph(f"个股估值分析报告", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"股票：{name}（{code}）", head_style))
    story.append(Paragraph(f"报告日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 15))

    if val_result:
        data = [
            ['指标', '数值'],
            ['当前价格', f"{val_result['current_price']:.2f} 元"],
            ['合理估值区间', f"{val_result['fair_low']:.2f} ~ {val_result['fair_high']:.2f} 元"],
            ['估值百分位', f"{val_result['percentile']:.1f}%"],
            ['相对估值', val_result['relative']],
            ['投资评级', val_result['rating']],
        ]
        t = Table(data, colWidths=[150, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,-1), CHINESE_FONT),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"建议：{val_result['advice']}", normal_style))
    else:
        story.append(Paragraph("估值数据不足，无法分析", normal_style))

    # 插入K线图（转为图片）
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
        stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
        stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
        fig.add_trace(go.Candlestick(x=stock_df['日期'], open=stock_df['开盘'], high=stock_df['最高'], low=stock_df['最低'], close=stock_df['收盘']), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA5'], name='MA5'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA10'], name='MA10'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA20'], name='MA20'), row=1, col=1)
        vol_colors = ['red' if o <= c else 'green' for o, c in zip(stock_df['开盘'], stock_df['收盘'])]
        fig.add_trace(go.Bar(x=stock_df['日期'], y=stock_df['成交量'], marker_color=vol_colors), row=2, col=1)
        fig.update_layout(height=400, template='plotly_white', xaxis_rangeslider_visible=False)
        img_bytes = pio.to_image(fig, format='png', width=600, height=400)
        img_buffer = io.BytesIO(img_bytes)
        from reportlab.platypus import Image
        img = Image(img_buffer, width=450, height=300)
        story.append(Spacer(1, 15))
        story.append(Paragraph("K线图", head_style))
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"K线图生成失败: {str(e)}", normal_style))

    doc.build(story)
    buf.seek(0)
    return buf

# ======================== Streamlit 主界面 ========================
def main():
    st.sidebar.markdown("<div class='sidebar-header'><h3>⚙️ 控制面板</h3></div>", unsafe_allow_html=True)

    # 数据获取按钮
    if st.sidebar.button("📥 获取/更新数据", type="primary"):
        with st.status("正在获取数据...", expanded=True) as status:
            progress_bar = st.progress(0, text="准备...")
            def update_progress(current, total, msg):
                progress_bar.progress(current/total, text=f"{current}/{total} {msg}")
            success = get_all_data(update_progress)
            if success:
                status.update(label="数据获取完成！", state="complete")
                st.success("数据已更新！")
                st.rerun()
            else:
                status.update(label="获取失败", state="error")

    # 加载数据
    df = load_local_data()
    if df is None or df.empty:
        st.info("👈 请先点击「获取/更新数据」加载行情")
        st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("📊 个股分析")
    stock_list = df['股票代码'].unique()
    stock_names = df.groupby('股票代码')['股票名称'].first().to_dict()
    selected = st.sidebar.selectbox("选择股票", stock_list, format_func=lambda x: f"{x} - {stock_names.get(x, '')}")

    # ========== 主页面 ==========
    st.title("📈 沪深300智能估值分析平台")
    st.caption("基于历史估值百分位与DCF模型，提供科学投资建议")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总记录数", len(df))
    col2.metric("个股数量", df['股票代码'].nunique())
    col3.metric("数据起始", df['日期'].min().strftime('%Y-%m-%d'))
    col4.metric("数据结束", df['日期'].max().strftime('%Y-%m-%d'))

    st.divider()

    # 个股分析
    stock_df = df[df['股票代码'] == selected].sort_values('日期').reset_index(drop=True)
    if stock_df.empty:
        st.warning("该股票暂无数据")
        return

    st.subheader(f"📌 {stock_names.get(selected, '')}（{selected}）")
    val = calculate_valuation(stock_df)

    if val:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前价格", f"{val['current_price']:.2f} 元")
        c2.metric("合理估值区间", f"{val['fair_low']:.2f} ~ {val['fair_high']:.2f} 元")
        c3.metric("估值百分位", f"{val['percentile']:.1f}%", delta=val['relative'])
        c4.metric("投资评级", val['rating'])

        st.markdown(f"""
        <div class="advice-box">
        <h4>💡 投资建议</h4>
        <p>{val['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

    # K线图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    stock_df['MA5'] = stock_df['收盘'].rolling(5).mean()
    stock_df['MA10'] = stock_df['收盘'].rolling(10).mean()
    stock_df['MA20'] = stock_df['收盘'].rolling(20).mean()
    fig.add_trace(go.Candlestick(x=stock_df['日期'], open=stock_df['开盘'], high=stock_df['最高'], low=stock_df['最低'], close=stock_df['收盘'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA5'], line=dict(color='blue'), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA10'], line=dict(color='orange'), name='MA10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['MA20'], line=dict(color='green'), name='MA20'), row=1, col=1)
    vol_colors = ['red' if o <= c else 'green' for o, c in zip(stock_df['开盘'], stock_df['收盘'])]
    fig.add_trace(go.Bar(x=stock_df['日期'], y=stock_df['成交量'], marker_color=vol_colors, name='成交量'), row=2, col=1)
    fig.update_layout(height=500, template='plotly_white', xaxis_rangeslider_visible=False, legend=dict(orientation='h', y=1.02))
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 估值指标走势
    st.subheader("📊 估值指标历史走势")
    if 'peTTM' in stock_df.columns:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['peTTM'], name='PE-TTM'), secondary_y=False)
        fig2.add_trace(go.Scatter(x=stock_df['日期'], y=stock_df['pbMRQ'], name='PB-MRQ'), secondary_y=True)
        fig2.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    # 导出PDF
    st.divider()
    if st.button("📄 导出PDF分析报告"):
        pdf_bytes = create_pdf_report(selected, stock_names.get(selected, ''), stock_df, val)
        st.download_button(
            label="⬇️ 下载报告",
            data=pdf_bytes,
            file_name=f"{selected}_估值报告_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()