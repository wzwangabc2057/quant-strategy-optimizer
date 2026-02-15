"""
================================================================================
多因子量化策略 v4 - 智能版
================================================================================
核心优化:
    1. 波动率调整动量 - 夏普式动量 (收益/波动)
    2. 智能止损 - ATR动态止损替代固定止损
    3. 精细反转加成分层
    4. 动量加成到权重倾斜
    5. 市场环境自适应

最佳结果: R4 33.43%, R5 36.40%, Sharpe 2.65/2.64
================================================================================
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BUY_COMMISSION, SELL_COMMISSION, PORTFOLIO_FILE,
    BACKTEST_START, BACKTEST_END, FACTOR_CONFIG,
    STABLE_WEIGHTS, AGGRESSIVE_WEIGHTS
)
from data.fetcher import DataFetcher


class ConfigV4:
    """v4增强配置"""
    BUY_COMMISSION = BUY_COMMISSION
    SELL_COMMISSION = SELL_COMMISSION

    # 波动率调整动量参数
    VOLATILITY_LOOKBACK = FACTOR_CONFIG.get('momentum_periods', {}).get('3m', {}).get('days', 60)
    MOMENTUM_PERIODS = FACTOR_CONFIG['momentum_periods']

    # 智能止损
    ATR_STOP_MULTIPLIER = FACTOR_CONFIG['atr_multiplier']
    ATR_PERIOD = FACTOR_CONFIG['atr_period']

    # 相关性
    MAX_CORRELATION = FACTOR_CONFIG['max_correlation']


def calculate_volatility_adjusted_momentum(price_series: pd.Series,
                                           periods: Dict = None) -> float:
    """波动率调整动量 (夏普式)"""
    if periods is None:
        periods = ConfigV4.MOMENTUM_PERIODS

    momentum_score = 0
    for name, config in periods.items():
        days = config['days']
        weight = config['weight']

        if len(price_series) < days:
            continue

        ret = price_series.iloc[-1] / price_series.iloc[-days] - 1
        vol = price_series.tail(days).pct_change().std() * np.sqrt(252)

        if vol > 0:
            risk_adj_ret = ret / vol
        else:
            risk_adj_ret = ret

        momentum_score += risk_adj_ret * weight

    return momentum_score


def apply_smart_stop_loss(position_return: float,
                          position_days: int,
                          atr_ratio: float,
                          max_loss: float = -0.20,
                          take_profit: float = 0.30) -> str:
    """智能止损 - 基于ATR和持仓时间"""
    if position_return >= take_profit:
        return 'take_profit'

    if position_return <= max_loss:
        return 'stop_loss'

    # ATR动态止损
    if position_days < 10:
        atr_stop = -atr_ratio * ConfigV4.ATR_STOP_MULTIPLIER * 0.75
    elif position_days < 30:
        atr_stop = -atr_ratio * ConfigV4.ATR_STOP_MULTIPLIER * 1.0
    else:
        atr_stop = -atr_ratio * ConfigV4.ATR_STOP_MULTIPLIER * 1.5

    if position_return <= atr_stop:
        return 'stop_loss'

    return 'hold'


def detect_market_regime(price_series: pd.Series) -> Tuple[str, float, float]:
    """检测市场环境 Returns: (regime, tilt_mult, max_weight_mult)"""
    if len(price_series) < 60:
        return 'volatile', 0.8, 0.8

    ret_20 = price_series.iloc[-1] / price_series.iloc[-20] - 1
    ret_60 = price_series.iloc[-1] / price_series.iloc[-60] - 1
    vol = price_series.pct_change().tail(20).std() * np.sqrt(252)

    ma20 = price_series.tail(20).mean()
    ma60 = price_series.tail(60).mean()
    trend = (ma20 / ma60 - 1)

    # 强牛市
    if ret_60 > 0.15 and ret_20 > 0.05 and vol < 0.25:
        return 'strong_bull', 1.4, 1.5

    # 温和牛市
    if ret_60 > 0.08 and trend > 0.02:
        return 'bull', 1.2, 1.2

    # 熊市
    if ret_60 < -0.15 and ret_20 < -0.05:
        return 'strong_bear', 0.4, 0.6

    if ret_60 < -0.08 or trend < -0.02:
        return 'bear', 0.6, 0.8

    # 高波动
    if vol > 0.35:
        return 'high_vol', 0.5, 0.7

    return 'neutral', 0.9, 0.9


def apply_nonlinear_weight_tilt(weights: pd.DataFrame,
                                scores: pd.DataFrame,
                                tilt_strength: float = 1.0,
                                max_weight: float = 0.08) -> pd.DataFrame:
    """非线性权重倾斜 - 带动量加成"""
    portfolio = weights[['code', 'weight']].copy()
    portfolio = portfolio.merge(scores[['code', 'composite_score', 'momentum_score']],
                                on='code', how='left')
    portfolio = portfolio.fillna({'composite_score': 50, 'momentum_score': 0})

    portfolio['percentile'] = portfolio['composite_score'].rank(pct=True)

    mom_std = portfolio['momentum_score'].std()
    if mom_std > 0:
        portfolio['momentum_z'] = (portfolio['momentum_score'] - portfolio['momentum_score'].mean()) / mom_std
    else:
        portfolio['momentum_z'] = 0

    def calc_multiplier(row):
        pct = row['percentile']
        mom_z = row['momentum_z']

        base = 1.0 + tilt_strength * (pct - 0.5)

        if pct >= 0.90:
            base *= 1.30
        elif pct >= 0.80:
            base *= 1.15
        elif pct >= 0.70:
            base *= 1.08
        elif pct < 0.10:
            base *= 0.70
        elif pct < 0.20:
            base *= 0.85
        elif pct < 0.30:
            base *= 0.92

        # 动量加成
        if mom_z > 1.0:
            base *= 1.05
        elif mom_z > 0.5:
            base *= 1.02
        elif mom_z < -1.0:
            base *= 0.95

        return base

    portfolio['multiplier'] = portfolio.apply(calc_multiplier, axis=1)
    portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']
    portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(upper=max_weight)

    total = portfolio['adjusted_weight'].sum()
    if total > 0:
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total

    return portfolio


def apply_reversal_boost(scores: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
    """反转加成 - 精细分层"""
    result = scores.copy()
    is_quality = (factor_data['roe'] > 5) & (factor_data.get('profit_growth', 0) > -10)

    if 'drawdown_3m' not in factor_data.columns:
        return result

    drawdown = -factor_data['drawdown_3m']
    boost = pd.Series(0, index=result.index)

    boost[is_quality & (drawdown > 30)] = 25
    boost[is_quality & (drawdown > 25) & (drawdown <= 30)] = 20
    boost[is_quality & (drawdown > 20) & (drawdown <= 25)] = 15
    boost[is_quality & (drawdown > 15) & (drawdown <= 20)] = 10
    boost[is_quality & (drawdown > 10) & (drawdown <= 15)] = 6
    boost[is_quality & (drawdown > 5) & (drawdown <= 10)] = 3

    result['composite_score'] = (result['composite_score'] + boost).clip(0, 100)

    return result


class StrategyV4:
    """v4智能策略"""

    def __init__(self, portfolio_file: str = None):
        self.fetcher = DataFetcher()
        self.portfolio_file = portfolio_file or PORTFOLIO_FILE

    def load_portfolio(self) -> dict:
        """加载持仓"""
        return self.fetcher.load_portfolio(self.portfolio_file)

    def build_factor_data(self, codes: list, price_pivot: pd.DataFrame,
                         eval_date: str) -> pd.DataFrame:
        """构建因子数据"""
        factor_df = pd.DataFrame({'code': codes})

        # 财务数据
        fin_df = self.fetcher.get_financial_data(codes)
        factor_df = factor_df.merge(fin_df, on='code', how='left')

        # 价格因子
        available_dates = [d for d in price_pivot.index if d <= eval_date]
        if len(available_dates) >= 120:
            recent = price_pivot.loc[available_dates].tail(120)

            # 多周期动量
            for period_name, config in ConfigV4.MOMENTUM_PERIODS.items():
                days = config['days']
                if len(recent) >= days:
                    ret = recent.iloc[-1] / recent.iloc[-days] - 1
                    factor_df[f'mom_{period_name}'] = factor_df['code'].map(ret.to_dict()).fillna(0)

            # 波动率调整动量
            mom_scores = {}
            for code in factor_df['code']:
                if code in recent.columns:
                    price_series = recent[code].dropna()
                    if len(price_series) >= 60:
                        mom_scores[code] = calculate_volatility_adjusted_momentum(price_series)
                    else:
                        mom_scores[code] = 0
            factor_df['momentum_score'] = factor_df['code'].map(mom_scores).fillna(0)

            # 回撤
            cummax = recent.cummax()
            drawdown = ((recent - cummax) / cummax).min() * 100
            factor_df['drawdown_3m'] = factor_df['code'].map(drawdown.to_dict()).fillna(-10)

            # 波动率
            vol = recent.pct_change().tail(60).std() * np.sqrt(252) * 100
            factor_df['volatility'] = factor_df['code'].map(vol.to_dict()).fillna(30)

        # 填充默认值
        defaults = {
            'roe': 10, 'eps': 1, 'profit_growth': 0,
            'momentum_score': 0, 'drawdown_3m': -10, 'volatility': 30
        }
        factor_df = factor_df.fillna(defaults)

        return factor_df

    def calculate_scores(self, factor_df: pd.DataFrame,
                        factor_weights: dict) -> pd.DataFrame:
        """计算因子得分"""
        result = factor_df.copy()

        def percentile_score(series, ascending=True):
            return series.rank(pct=True) * 100 if ascending else (1 - series.rank(pct=True)) * 100

        if 'roe' in factor_df.columns:
            result['roe_score'] = percentile_score(factor_df['roe'], True)
        if 'momentum_score' in factor_df.columns:
            result['momentum_score_raw'] = factor_df['momentum_score']
            result['momentum_score'] = percentile_score(factor_df['momentum_score'], True)
        if 'drawdown_3m' in factor_df.columns and 'roe' in factor_df.columns:
            result['reversal_score'] = percentile_score(-factor_df['drawdown_3m'], True)
        if 'volatility' in factor_df.columns:
            result['low_volatility_score'] = percentile_score(factor_df['volatility'], False)

        # 综合得分
        composite = pd.Series(50.0, index=result.index)
        for factor, weight in factor_weights.items():
            score_col = f'{factor}_score'
            if score_col in result.columns:
                composite += (result[score_col].fillna(50) - 50) * weight * 2

        result['composite_score'] = composite.clip(0, 100)

        return result

    def run_backtest(self, portfolio: pd.DataFrame, price_pivot: pd.DataFrame,
                    strategy_type: str = 'stable',
                    start_date: str = None, end_date: str = None) -> dict:
        """运行回测"""
        factor_weights = STABLE_WEIGHTS if strategy_type == 'stable' else AGGRESSIVE_WEIGHTS
        start_date = start_date or BACKTEST_START
        end_date = end_date or BACKTEST_END

        codes = portfolio['code'].tolist()
        dates = sorted([d for d in price_pivot.index if start_date <= d <= end_date])

        market_avg = price_pivot.mean(axis=1)

        # 月度调仓
        rebalance_dates = []
        current_month = None
        for d in dates:
            ym = d[:7]
            if ym != current_month:
                current_month = ym
                rebalance_dates.append(d)

        # 初始化
        current_weights = dict(zip(portfolio['code'], portfolio['weight']))
        current_weights = {k: v/sum(current_weights.values()) for k, v in current_weights.items()}
        original_weights = current_weights.copy()

        position_tracker = {code: {'cost': None, 'days': 0, 'atr': 0} for code in codes}

        enhanced_returns = []
        original_returns = []
        regime_log = []

        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]

            # 调仓
            if date in rebalance_dates and i > 1:
                market_up_to_now = market_avg.loc[:date]
                regime, tilt_mult, weight_mult = detect_market_regime(market_up_to_now)

                actual_tilt = 1.0 * tilt_mult
                actual_max_weight = 0.08 * weight_mult

                factor_df = self.build_factor_data(codes, price_pivot, date)
                scored_df = self.calculate_scores(factor_df, factor_weights)
                scored_df = apply_reversal_boost(scored_df, factor_df)

                enhanced_portfolio = apply_nonlinear_weight_tilt(
                    portfolio, scored_df,
                    tilt_strength=actual_tilt,
                    max_weight=actual_max_weight
                )

                new_weights = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))

                turnover = sum(abs(new_weights.get(c, 0) - current_weights.get(c, 0)) for c in codes) / 2
                trade_cost = turnover * (ConfigV4.BUY_COMMISSION + ConfigV4.SELL_COMMISSION)

                current_weights = new_weights

                for code in codes:
                    if code in price_pivot.columns:
                        current_price = price_pivot.loc[date, code]
                        if pd.notna(current_price):
                            position_tracker[code]['cost'] = current_price
                            position_tracker[code]['days'] = 0
                            recent = price_pivot.loc[:date, code].tail(ConfigV4.ATR_PERIOD)
                            if len(recent) >= ConfigV4.ATR_PERIOD:
                                atr = recent.pct_change().std() * current_price
                                position_tracker[code]['atr'] = atr / current_price if current_price > 0 else 0.03

                regime_log.append({
                    'date': date, 'regime': regime,
                    'turnover': turnover, 'trade_cost': trade_cost
                })

            # 计算收益
            enh_ret = 0
            orig_ret = 0

            for code in codes:
                if code not in price_pivot.columns:
                    continue

                curr_price = price_pivot.loc[date, code]
                prev_price = price_pivot.loc[prev_date, code]

                if pd.isna(curr_price) or pd.isna(prev_price) or prev_price <= 0:
                    continue

                stock_ret = curr_price / prev_price - 1

                position_tracker[code]['days'] += 1

                # 智能止损
                pos_info = position_tracker[code]
                if pos_info['cost'] is not None:
                    pos_return = curr_price / pos_info['cost'] - 1
                    stop_action = apply_smart_stop_loss(
                        pos_return, pos_info['days'], pos_info['atr']
                    )
                    if stop_action in ['stop_loss', 'take_profit']:
                        current_weights[code] = 0

                enh_ret += stock_ret * current_weights.get(code, 0)
                orig_ret += stock_ret * original_weights.get(code, 0)

            if regime_log and date == regime_log[-1]['date']:
                enh_ret -= regime_log[-1]['trade_cost']

            enhanced_returns.append({'date': date, 'return': enh_ret})
            original_returns.append({'date': date, 'return': orig_ret})

        # 计算指标
        def calc_metrics(daily_rets):
            rets = np.array([r['return'] for r in daily_rets])
            cum = np.cumprod(1 + rets)
            total_return = cum[-1] - 1
            n_years = len(rets) / 252
            annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_drawdown = dd.min()

            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

            return {
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'total_return': total_return
            }

        enhanced_metrics = calc_metrics(enhanced_returns)
        original_metrics = calc_metrics(original_returns)

        return {
            'enhanced': enhanced_metrics,
            'original': original_metrics,
            'improvement': enhanced_metrics['annual_return'] - original_metrics['annual_return'],
        }


def main():
    print("=" * 70)
    print("多因子量化策略 v4 - 智能版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    strategy = StrategyV4()
    portfolio = strategy.load_portfolio()

    r4 = portfolio['r4']
    r5 = portfolio['r5']

    print(f"\nR4稳健型: {len(r4)} 只股票")
    print(f"R5进取型: {len(r5)} 只股票")

    # 获取价格
    all_codes = list(set(r4['code'].tolist() + r5['code'].tolist()))
    print("\n获取价格数据...")
    price_df = strategy.fetcher.get_prices(all_codes, '2019-01-01', '2025-12-31')
    price_pivot = price_df.pivot(index='date', columns='code', values='close')
    print(f"价格数据: {len(price_pivot)} 个交易日, {len(price_pivot.columns)} 只股票")

    # R4
    print("\n" + "=" * 50)
    print("【R4 稳健型】")
    r4_result = strategy.run_backtest(r4, price_pivot, 'stable')
    r4_enh, r4_orig = r4_result['enhanced'], r4_result['original']
    print(f"  原始: 年化 {r4_orig['annual_return']*100:.2f}% | 夏普 {r4_orig['sharpe']:.2f}")
    print(f"  v4:   年化 {r4_enh['annual_return']*100:.2f}% | 夏普 {r4_enh['sharpe']:.2f}")
    print(f"  提升: {r4_result['improvement']*100:+.2f}%")

    # R5
    print("\n" + "=" * 50)
    print("【R5 进取型】")
    r5_result = strategy.run_backtest(r5, price_pivot, 'aggressive')
    r5_enh, r5_orig = r5_result['enhanced'], r5_result['original']
    print(f"  原始: 年化 {r5_orig['annual_return']*100:.2f}% | 夏普 {r5_orig['sharpe']:.2f}")
    print(f"  v4:   年化 {r5_enh['annual_return']*100:.2f}% | 夏普 {r5_enh['sharpe']:.2f}")
    print(f"  提升: {r5_result['improvement']*100:+.2f}%")

    print("\n" + "=" * 70)
    print("【v4 智能版结果汇总】")
    print("=" * 70)
    print(f"| 组合   | 年化收益 | 最大回撤 | 夏普比率 |")
    print(f"|--------|----------|----------|----------|")
    print(f"| R4稳健 | {r4_enh['annual_return']*100:>7.2f}% | {r4_enh['max_drawdown']*100:>7.1f}% | {r4_enh['sharpe']:>8.2f} |")
    print(f"| R5进取 | {r5_enh['annual_return']*100:>7.2f}% | {r5_enh['max_drawdown']*100:>7.1f}% | {r5_enh['sharpe']:>8.2f} |")


if __name__ == '__main__':
    main()
