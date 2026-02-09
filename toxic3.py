# -*- coding: utf-8 -*-
"""

create time is 2025.



@author: dhhly
"""
import warnings
import os
import pandas as pd
import numpy as np
import datetime
import sys
warnings.filterwarnings('ignore')




sz50_df = pd.read_csv('sz50_english_name.csv',encoding = 'GB18030')
sz50_df['code'] = sz50_df['代码'].apply(lambda x:x[:6])

# 只处理指定股票（可选）：
# - 设置环境变量 TARGET_CODES="600519,600000,600036"
# - 或设置 CODE_LIST=路径（文本/CSV，每行一个代码）
target_codes_env = os.environ.get('TARGET_CODES')
code_list_path = os.environ.get('CODE_LIST')
selected_codes = None
if code_list_path and os.path.exists(code_list_path):
    try:
        with open(code_list_path, 'r', encoding='utf-8') as f:
            selected_codes = [line.strip()[:6] for line in f if line.strip()]
    except Exception:
        selected_codes = None
elif target_codes_env:
    selected_codes = [c.strip()[:6] for c in target_codes_env.split(',') if c.strip()]

if selected_codes:
    sz50_df = sz50_df[sz50_df['code'].isin(selected_codes)].copy()

all_df = pd.DataFrame()
sum_info = pd.DataFrame()


all_buy_df = pd.DataFrame()
all_sell_df = pd.DataFrame()

all_can = pd.DataFrame()


i = 0
for index,row in sz50_df.iterrows():
    
    
    i=i+1
    code = row['code']
    name = row['name']
    chinese_name = row['简称']
    print(i,code,name,chinese_name)
    taq_file = 'SHL2_TAQ_'+code+'_201912.csv'
    data = pd.read_csv(taq_file,header=None) #茅台 注意上海的格式跟深圳

    data['time']=data[2]
    data['best ask']=data[19]
    data['best ask qty'] = data[40]
    data['best bid'] = data[20]
    data['best bid qty'] = data[41]
    data['daily accumulated vol'] = data[52]
    
    data['date'] = data[1]
    d=data[['time','best bid','best ask','best bid qty','best ask qty','daily accumulated vol']]
    d['vol shift1']=d['daily accumulated vol'].shift(1)
    
    acc_vol= d[(d['daily accumulated vol']==0) & (d['vol shift1']>0) ] #这样计算忽略了最后一天的情况
    sum_vol = acc_vol['vol shift1'].sum()+d['daily accumulated vol'].iloc[-1]
    
    d['time']=pd.to_datetime(d['time'])
    
    d = d[d['best ask']>d['best bid']]  #这一步相当于把集合竞价前的时间都剔除掉了。
    
    #d= d[d['best ask']-d['best bid']<0.002]
    d['mid'] = (d['best bid']+d['best ask'])/2
    
            
    # can这个变量之前在 weighted 拼接 trade ob信息。
    
    # can=d.set_index('time',drop = False)  
    
    can = d
    

    
    
    #下面是关于can的处理为了拿到1分钟过后的weighted middle priced
    # --- 1) 计算 Weighted Mid Price (WMP) ---
    # WMP = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
    can = can.sort_values('time').copy()
    denom = can['best bid qty'] + can['best ask qty']
    can['WMP'] = (can['best bid'] * can['best ask qty'] + can['best ask'] * can['best bid qty']) / denom
    # 若偶发 denom=0（极少见），置 NaN
    can.loc[denom == 0, 'WMP'] = np.nan
    
    
    # sys.exit()
    
    # --- 2) 标记交易日与时段（仅连续交易时段才有效） ---
    # 使用归一化到当天 00:00:00 的 Timestamp，避免后续出现 python 的 datetime.date（object）类型
    can['date'] = can['time'].dt.normalize()
    tod = can['time'].dt.time
    
    # 定义两个连续交易时段
    AM_START, AM_END = pd.to_datetime('09:30:00').time(), pd.to_datetime('11:30:00').time()
    PM_START, PM_END = pd.to_datetime('13:00:00').time(), pd.to_datetime('14:57:00').time()
    
    def session_of(t):
        if AM_START <= t < AM_END:
            return 'AM'
        if PM_START <= t < PM_END:
            return 'PM'
        return None
    
    can['session'] = [session_of(t) for t in tod]
    
    
    # 开盘30分钟：09:30:00 - 10:00:00
    OPEN30_START, OPEN30_END = AM_START, pd.to_datetime('10:00:00').time()
    # 收盘前30分钟：14:27:00 - 14:57:00
    CLOSE30_START, CLOSE30_END = pd.to_datetime('14:30:00').time(), PM_END
    
    t = can['time'].dt.time
    
    is_open30  = (t >= OPEN30_START)  & (t < OPEN30_END)
    is_close30 = (t >= CLOSE30_START) & (t < CLOSE30_END)
    # “中间”= 连续竞价时段内，且不属于 open30/close30
    in_continuous = can['session'].isin(['AM','PM'])
    is_mid = in_continuous & (~is_open30) & (~is_close30)
    
    # 三个0/1哑变量
    # 避免与后面的价格中位数列 'mid' 同名冲突，这里使用 is_* 命名
    can['is_open30']  = is_open30.astype(int)
    can['is_mid']     = is_mid.astype(int)
    can['is_close30'] = is_close30.astype(int)
    
    # 也给一个互斥的分类标签，便于分组画图/做回归FE
    can['tod_bucket'] = np.select(
        [is_open30, is_mid, is_close30],
        ['OPEN30', 'MID', 'CLOSE30'],
        default='OUT'  # 非连续竞价时间（一般会被后续过滤）
        )
    
    
    
    
    # 只在连续交易时段保留 WMP（可选：不想置空就跳过这一步）
    # can.loc[can['session'].isna(), 'WMP'] = np.nan
    
    # --- 3) 计算 1 分钟后的目标时间，并在时段边界外置 NaN ---
    can['target_time'] = can['time'] + pd.Timedelta(minutes=1)
    
    # 当前行所属时段的结束时间（同一天）
    end_time_map = {'AM': AM_END, 'PM': PM_END}
    # 构造该行对应的“当日时段结束的 Timestamp”：直接用归一化日期 + 当日时间偏移
    can['session_end_ts'] = pd.NaT
    mask_am = can['session'].eq('AM')
    mask_pm = can['session'].eq('PM')
    can.loc[mask_am, 'session_end_ts'] = can.loc[mask_am, 'date'] + pd.to_timedelta('11:30:00')
    can.loc[mask_pm, 'session_end_ts'] = can.loc[mask_pm, 'date'] + pd.to_timedelta('14:57:00')
    
    # 只有在连续交易时段，且 t+60s 未越过这一时段的结束，才允许取到 WMP_1m
    valid_rows = can['session'].notna() & (can['target_time'] < can['session_end_ts'])
    
    # --- 4) 用 asof join 找到 “目标时间之后的第一条快照” 作为 1 分钟后的 WMP ---
    # 先准备一个查找表（只包含连续交易时段的快照，避免跨时段/跨日）
    lookup = can.loc[can['session'].notna(), ['date', 'time', 'WMP']].rename(columns={'time':'snap_time'}).copy()
    lookup = lookup.sort_values(['date','snap_time'])
    
    # 左表：仅对 valid_rows 计算 WMP_1m，其余设为 NaN
    left = can.loc[valid_rows, ['date', 'time', 'target_time']].rename(columns={'target_time':'query_time'}).copy()
    left = left.sort_values(['date','query_time'])
    
    # asof 需要按时间排序，并通过 by= 分组以避免跨日匹配
    wmp_1m = pd.merge_asof(
        left,
        lookup,
        left_on='query_time',
        right_on='snap_time',
        by='date',
        direction='forward',          # 目标是 t+60s 之后“最靠近的快照”
        tolerance=pd.Timedelta(seconds=6)  # 快照是3秒一次，这里给个6秒容差更稳
    )
    
    # 把匹配到的 WMP_1m 回填到 can
    can['WMP_1m'] = np.nan
    can.loc[valid_rows, 'WMP_1m'] = wmp_1m['WMP'].values
    
    # --- 5) 清理与可选派生 ---
    # 也可以顺手算 spread/OBI 等慢变量（以后与 trade 合并时会用到）
    can['spread'] = can['best ask'] - can['best bid']
    can['mid'] = (can['best ask'] + can['best bid']) / 2
    can['obi'] = (can['best bid qty'] - can['best ask qty']) / (can['best bid qty'] + can['best ask qty'])
    
    # 可选：如果你更倾向“直接剔除”边界噪声样本，可以在后续建模前：
    # can = can[can['WMP_1m'].notna()].copy()
    
    
    # 用时间索引做基于“秒”的 rolling
    can = can.sort_values('time').copy()
    can_idx = can.set_index('time')
    
    def add_rv_and_mom(df):
        s = df.sort_index().copy()
        s['logwmp'] = np.log(s['WMP'])
        s['ret'] = s['logwmp'].diff()
        # 60s 窗口的时间滚动标准差（pandas按时间窗滚动）
        s['rv_60s'] = s['ret'].rolling('60s').std()
        # 2) 动量（10s 时间位移的价差，可换成比例动量：pct_change(freq='10s')）
        # s['mom_10s'] = s['WMP'] - s['WMP'].shift(freq='10s')
        s['mom_60s_pct'] = s['WMP'].pct_change(freq='60s')
        return s
    
    am = can_idx.loc[can_idx['session'].eq('AM')].pipe(add_rv_and_mom)
    pm = can_idx.loc[can_idx['session'].eq('PM')].pipe(add_rv_and_mom)
    can_idx = pd.concat([am, pm]).sort_index()
    
    # 回填回 can（去掉临时列）
    can = can_idx.reset_index()
    can.rename(columns={'index':'time'}, inplace=True)
    
    # sys.exit()
    
    
    
    trading_days = can['time'].dt.day.unique()

    transaction_file = 'SHL2_TRANSACTION_'+code+'_201912.csv'
    trade= pd.read_csv(transaction_file,header=None)
    trade = trade[[1,5,6,8,12]]
    trade.columns = ['TRD','TRDPRICE','TRDQTY','UNIX','BSFLAG'] #为了对应之前做回测用的，实际上原始数据没有这些，只是方便复用load_shanghai_trade_20220318.py
    
    
    
    trade['time']=datetime.datetime(1970,1,1)+trade['UNIX'].apply(
            lambda x: datetime.timedelta(milliseconds=x))+datetime.timedelta(hours=8)
    trade['BSFLAG']=np.where(trade['BSFLAG']=='B',1,-1)
    trade['trade'] = trade['TRDQTY']*trade['BSFLAG']
    
    # trade=trade.set_index('time',drop=False)
    
    
    


    # -------------------- 导出 Hawkes 1D 事件文件（每只股票） --------------------
    # 可选：通过环境变量 DAY=YYYYMMDD 仅保留该日，否则默认取首个交易日
    day_env = os.environ.get('DAY')
    trade_ts = trade[['time']].copy()
    if day_env and len(day_env) == 8:
        d0 = pd.to_datetime(day_env, format='%Y%m%d')
        d1 = d0 + pd.Timedelta(days=1)
        trade_ts = trade_ts[(trade_ts['time'] >= d0) & (trade_ts['time'] < d1)].copy()
    else:
        if not trade_ts.empty:
            first_day = trade_ts['time'].dt.normalize().iloc[0]
            next_day = first_day + pd.Timedelta(days=1)
            trade_ts = trade_ts[(trade_ts['time'] >= first_day) & (trade_ts['time'] < next_day)].copy()

    tt = trade_ts['time'].dt.time
    in_am = (tt >= pd.to_datetime('09:30:00').time()) & (tt < pd.to_datetime('11:30:00').time())
    in_pm = (tt >= pd.to_datetime('13:00:00').time()) & (tt < pd.to_datetime('14:57:00').time())
    trade_ts = trade_ts[in_am | in_pm].copy()
    sec = (trade_ts['time'].dt.hour*3600 + trade_ts['time'].dt.minute*60 + trade_ts['time'].dt.second \
           + trade_ts['time'].dt.microsecond/1e6).sort_values().to_numpy()
    events_json = [{"t": float(x)} for x in sec]
    out_events = f"events_{code}.json"
    import json as _json
    with open(out_events, 'w', encoding='utf-8') as _f:
        _json.dump(events_json, _f, ensure_ascii=False)
    with open(f"T_{code}.txt", 'w', encoding='utf-8') as _ft:
        _ft.write(str(float(sec.max()) if len(sec)>0 else 0.0))
    prior_b1 = 2*np.pi/14220.0
    print(f"[events] {code}: {len(sec)} events saved -> {out_events}; prior b1≈{prior_b1:.6g}")

    #尝试合并trade跟ob快照    
    trade_ag = trade.reset_index()
    can_ag = can.reset_index()
    # merge_asof 要求按键排序
    trade_ag = trade_ag.sort_values('time')
    can_ag = can_ag.sort_values('time')
    
    trade_ag = trade_ag.loc[:, ~trade_ag.columns.duplicated()]
    can_ag = can_ag.loc[:, ~can_ag.columns.duplicated()]    
    
    # 用 merge_asof 实现 "每笔交易匹配最近一个快照"
    combine = pd.merge_asof(
        trade_ag,
        can_ag,
        on='time',
        direction='backward'  )
    
    
    combine['trade impact'] = combine['WMP_1m']-combine['TRDPRICE']
    combine['market impact'] = combine['WMP_1m'] - combine['WMP']
    
    # 只保留能匹配到 1 分钟后 WMP 的交易（避免靠近时段边界的 NaN）
    combine = combine[combine['WMP_1m'].notna()].copy()
    
    # 成交方向 s（买=+1，卖=-1）
    combine['s'] = combine['BSFLAG'].astype(int)

    # 一档对手深度（机制检验会用到）
    combine['opp_depth'] = np.where(combine['s'] == 1, combine['best ask qty'], combine['best bid qty'])   #注意这个可能不一定对，因为是3秒一个截屏，有可能中间变了，这个作为保留吧
    
    # 单边有效价差 OES = s*(P - WMP)
    combine['OES'] = combine['s'] * (combine['TRDPRICE'] - combine['WMP'])
    
    # 统一定义的 Trade Impact / Market Impact（方向对齐）
    combine['TI'] = combine['s'] * (combine['WMP_1m'] - combine['TRDPRICE'])
    combine['MI'] = combine['s'] * (combine['WMP_1m'] - combine['WMP'])   # 检查恒等式：应当 MI ≈ TI + OES
    # 可加一列校验误差（微小数值误差正常）
    combine['check_MI_eq_TI_plus_OES'] = combine['MI'] - (combine['TI'] + combine['OES'])
    
    # 标准化成 bp（相对于当下 WMP）
    combine['TI_bp']  = combine['TI']  / combine['WMP'] * 1e4
    combine['OES_bp'] = combine['OES'] / combine['WMP'] * 1e4
    combine['MI_bp']  = combine['MI']  / combine['WMP'] * 1e4
    
    # 状态变量也带下去（已经通过 asof 对齐）：spread, obi, rv_60s, mom_10s, best bid/ask qty 等都在 combine 里
    # 你也可以构造“薄厚”指标，便于回归：如 depth 对数
    combine['log_opp_depth'] = np.log1p(combine['opp_depth'])


    buy = combine[combine['BSFLAG']==1]
    sell = combine[combine['BSFLAG']==-1]
    
    
    
    
    
    print('buy trade impact:',buy['trade impact'].mean())
    print('sell trade impact:',sell['trade impact'].mean())
    print('buy market impact:',buy['market impact'].mean())
    print('sell market impact:',sell['market impact'].mean())
    
    
    
    
    
    buy_average_trade_impact = -buy['trade impact'].mean()
    sell_average_trade_impact = sell['trade impact'].mean()
    average_price = can['WMP'].mean()
    average_queue = ((can['best bid qty']+can['best ask qty']     )/2).mean()
    average_spread = can['spread'].mean()
    average_relative_spread = average_spread/average_price
    
    buy_trade_impact_adjusted_by_price = -buy['trade impact']/buy['WMP']
    sell_trade_impact_adjusted_by_price = sell['trade impact']/sell['WMP']
    
    buy_trade_impact_adjusted_by_spread = -buy['trade impact']/can['spread'].mean()   #我觉得在计算这个spread不应该看当前的spread，跳跃性会很大
    sell_trade_impact_adjusted_by_spread = sell['trade impact']/can['spread'].mean()

    
    average_buy_trade_impact_adjusted_by_price = buy_trade_impact_adjusted_by_price.mean()
    average_buy_trade_impact_adjusted_by_spread = buy_trade_impact_adjusted_by_spread.mean()
    average_sell_trade_impact_adjusted_by_price = sell_trade_impact_adjusted_by_price.mean()
    average_sell_trade_impact_adjusted_by_spread = sell_trade_impact_adjusted_by_spread.mean()


    
    security_df = pd.DataFrame(index = [code],data=[[name,chinese_name,buy_average_trade_impact,sell_average_trade_impact,average_price,average_queue,
                                                     average_spread,average_relative_spread,
                                                     average_buy_trade_impact_adjusted_by_price,average_buy_trade_impact_adjusted_by_spread,
                                                     average_sell_trade_impact_adjusted_by_price,average_sell_trade_impact_adjusted_by_spread
                                                     
                                                     
                                                     ]])
    
    security_df.columns = ['name','chinese_name','buy_average_trade_impact','sell_average_trade_impact','average_price','average_queue',
                               'average_spread','average_relative_spread',
                               'average_buy_ti_by_price','average_buy_ti_by_spread',
                               'average_sell_ti_by_price','average_sell_timpact_by_spread'
                               
                              ]    
    
    
    all_df = pd.concat([all_df,security_df])
    
    all_buy_df = pd.concat([all_buy_df,buy])
    all_sell_df = pd.concat([all_sell_df,sell])

    all_can = pd.concat([all_can,can])
    
    # break
    # if i >3:
    #     break
    # break


all_can_corr = all_can.corr(numeric_only=True)

all_df_corr = all_df.corr(numeric_only=True)


all_buy_df['trade impact o'] = -all_buy_df['trade impact']
all_sell_df['trade impact o'] = all_sell_df['trade impact']

all_buy_df['ti by p'] = all_buy_df['trade impact o']/all_buy_df['WMP']
all_sell_df['ti by p'] = all_sell_df['trade impact o']/all_sell_df['WMP']


all_buy_df['queue'] = (all_buy_df['best bid qty']+all_buy_df['best ask qty'])/2
all_sell_df['queue'] = (all_sell_df['best bid qty']+all_sell_df['best ask qty'])/2

all_buy_df['re_spread'] = all_buy_df['spread']/all_buy_df['WMP']
all_sell_df['re_spread'] = all_sell_df['spread']/all_sell_df['WMP']



# all_df.to_csv('all_df.csv')
# all_df_corr.to_csv('all_df_corr.csv')


def grouped_stats(df, group_var, var='ti by p', q=3, side='buy'):
    """对某个状态变量分组，计算限单利润均值，返回DataFrame"""
    if pd.api.types.is_numeric_dtype(df[group_var]):
        labels = [f'Q{i+1}' for i in range(q)]
        df[group_var+'_bin'] = pd.qcut(df[group_var], q=q, labels=labels, duplicates='drop')
        g = df.groupby(group_var+'_bin')[var].mean().reset_index()
        g.rename(columns={group_var+'_bin': 'bin', var: 'mean_ti_by_p'}, inplace=True)
    else:
        g = df.groupby(group_var)[var].mean().reset_index()
        g.rename(columns={group_var: 'bin', var: 'mean_ti_by_p'}, inplace=True)
    g['feature'] = group_var
    g['side'] = side
    return g

# 买限单
buy_obi   = grouped_stats(all_buy_df, 'obi', var='ti by p', q=3, side='buy')
buy_mom   = grouped_stats(all_buy_df, 'mom_60s_pct', var='ti by p', q=3, side='buy')
buy_vol   = grouped_stats(all_buy_df, 'rv_60s', var='ti by p', q=3, side='buy')
buy_tod   = grouped_stats(all_buy_df, 'tod_bucket', var='ti by p', side='buy')

buy_p  = grouped_stats(all_buy_df, 'WMP', var='ti by p', side='buy')
buy_q  = grouped_stats(all_buy_df, 'queue', var='ti by p', side='buy')
buy_spread =  grouped_stats(all_buy_df, 'spread', var='ti by p', side='buy')
buy_re_spread =  grouped_stats(all_buy_df, 're_spread', var='ti by p', side='buy')


# 卖限单
sell_obi  = grouped_stats(all_sell_df, 'obi', var='ti by p', q=3, side='sell')
sell_mom  = grouped_stats(all_sell_df, 'mom_60s_pct', var='ti by p', q=3, side='sell')
sell_vol  = grouped_stats(all_sell_df, 'rv_60s', var='ti by p', q=3, side='sell')
sell_tod  = grouped_stats(all_sell_df, 'tod_bucket', var='ti by p', side='sell')

sell_p  = grouped_stats(all_sell_df, 'WMP', var='ti by p', side='sell')
sell_q  = grouped_stats(all_sell_df, 'queue', var='ti by p', side='sell')
sell_spread  = grouped_stats(all_sell_df, 'spread', var='ti by p', side='sell')
sell_re_spread  = grouped_stats(all_sell_df, 're_spread', var='ti by p', side='sell')


# 合并所有结果

# results_df = pd.concat([buy_obi, buy_mom, buy_vol, buy_tod,
#                         sell_obi, sell_mom, sell_vol, sell_tod],
#                        ignore_index=True)

results_df = pd.concat([buy_obi, buy_mom, buy_vol, buy_tod,buy_p,buy_q,buy_spread,buy_re_spread,
                        sell_obi, sell_mom, sell_vol, sell_tod,sell_p,sell_q,sell_spread,sell_re_spread],
                       ignore_index=True)




# results_df.to_csv('feature.csv')


# -------------------- 导出明细数据 --------------------
def to_utc_iso(shanghai_time_series):
    t = pd.to_datetime(shanghai_time_series, errors='coerce')
    # 本地时间为上海时区，转为 UTC ISO 字符串
    t = t.dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
    return t.dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

# 需要的列（若不存在则忽略）
cols_needed = [
    'time','BSFLAG','TRDQTY','TRDPRICE','WMP','WMP_1m',
    'TI_bp','MI_bp','OES_bp','spread','obi','rv_60s','mom_60s_pct','tod_bucket','opp_depth'
]

def export_subset(df, cols, out_path):
    present = [c for c in cols if c in df.columns]
    sub = df[present].copy()
    if 'time' in sub.columns:
        sub['time'] = to_utc_iso(sub['time'])
    sub.to_csv(out_path, index=False)

export_subset(all_buy_df, cols_needed, 'all_buy_df.csv')
export_subset(all_sell_df, cols_needed, 'all_sell_df.csv')

# 快照数据导出（不含成交列）
can_cols = ['time','WMP','WMP_1m','spread','obi','rv_60s','mom_60s_pct','tod_bucket']
export_subset(all_can, can_cols, 'all_can.csv')
        
