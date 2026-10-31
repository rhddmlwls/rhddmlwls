import unittest

import numpy as np
import pandas as pd

from trading.directional import simulate_directional


def bars(opening,closing):
    return pd.DataFrame(dict(open=opening,close=closing,
        high=np.maximum(opening,closing)*1.01,low=np.minimum(opening,closing)*.99),
        index=pd.date_range("2021-01-01",periods=len(opening),freq="D",tz="UTC"))


class DirectionalExecutionTests(unittest.TestCase):
    def test_all_cash_has_reusable_empty_ledger_schema(self):
        result=simulate_directional(bars([100.,100.],[100.,100.]),[0,0])
        self.assertEqual(result.stats['net_pnl'],0.)
        self.assertIn('net_pnl',result.positions.columns)
        self.assertIn('datetime',result.orders.columns)

    def test_short_sizing_next_open_and_no_daily_rebalancing(self):
        b=bars([100.,100.,80.],[100.,90.,85.])
        result=simulate_directional(b,[-1,0,0],fees=0,slippage=0)
        np.testing.assert_allclose(result.curve.equity,[10000.,11000.,12000.])
        self.assertEqual(result.orders.datetime.tolist(),b.index[1:].tolist())
        self.assertEqual(result.stats['short_pnl'],2000.)
        self.assertEqual(result.stats['long_pnl'],0.)
        self.assertEqual(result.stats['positions'],1)

    def test_short_funding_signs_and_open_boundary(self):
        b=bars([100.]*4,[100.]*4)
        events=pd.DataFrame(dict(datetime=[b.index[1],b.index[1]+pd.Timedelta('8h'),b.index[2]],
            last_funding_rate=[.01,.01,-.02],mark_price=[150.]*3))
        result=simulate_directional(b,[-1,0,0,0],events,fees=0,slippage=0)
        np.testing.assert_allclose(result.curve.funding_cost,[0.,-150.,300.,0.])
        self.assertAlmostEqual(result.stats['final_equity'],9850.)
        self.assertAlmostEqual(result.positions.funding_cost.iloc[0],150.)

    def test_reversal_closes_before_resizing_and_charges_both_legs(self):
        b=bars([100.]*4,[100.]*4)
        c=.003
        result=simulate_directional(b,[1,-1,0,0],fees=.001,slippage=.002)
        first_notional=10000/(1+c)
        after_long_exit=10000-2*c*first_notional
        second_notional=after_long_exit/(1+c)
        expected=after_long_exit-2*c*second_notional
        self.assertEqual(result.orders.action.tolist(),['entry','exit','entry','exit'])
        self.assertEqual(result.orders.side.tolist(),['buy','sell','sell','buy'])
        self.assertAlmostEqual(result.orders.notional.iloc[2],second_notional)
        self.assertAlmostEqual(result.stats['final_equity'],expected)
        self.assertAlmostEqual(result.stats['transaction_cost'],10000-expected)

    def test_terminal_close_includes_fee_and_open_pnl(self):
        b=bars([100.,100.,90.],[100.,90.,80.])
        result=simulate_directional(b,[-1,-1,-1],fees=.001,slippage=0)
        qty=10000/(100*1.001)
        expected=10000+qty*20-qty*100*.001-qty*80*.001
        self.assertAlmostEqual(result.stats['final_equity'],expected)
        self.assertEqual(result.positions.status.tolist(),['final_exit'])
        self.assertEqual(result.curve.quantity.iloc[-1],0.)
        self.assertEqual(result.orders.action.tolist(),['entry','final_exit'])


    def test_long_and_short_ledgers_sum_to_equity_pnl(self):
        b=bars([100.,100.,110.,90.,80.],[100.,105.,100.,85.,85.])
        result=simulate_directional(b,[1,-1,0,1,1],fees=.0005,slippage=.0005)
        self.assertAlmostEqual(result.positions.net_pnl.sum(),result.stats['net_pnl'])
        self.assertAlmostEqual(result.stats['long_pnl']+result.stats['short_pnl'],result.stats['net_pnl'])
        self.assertAlmostEqual(result.curve.long_pnl.sum(),result.stats['long_pnl'])
        self.assertAlmostEqual(result.curve.short_pnl.sum(),result.stats['short_pnl'])

    def test_future_signal_cannot_change_earlier_equity(self):
        b=bars([100.,100.,110.,90.,80.],[100.,105.,100.,85.,85.])
        a=simulate_directional(b,[1,-1,0,1,1],close_final=False)
        changed=simulate_directional(b,[1,-1,1,-1,-1],close_final=False)
        np.testing.assert_allclose(a.curve.equity.iloc[:3],changed.curve.equity.iloc[:3])

    def test_intrabar_short_exhaustion_is_rejected(self):
        b=bars([100.,100.],[100.,100.])
        b.iloc[1,b.columns.get_loc('high')]=201.
        with self.assertRaisesRegex(RuntimeError,'Intrabar equity'):
            simulate_directional(b,[-1,0],fees=0,slippage=0)


if __name__=='__main__':
    unittest.main()
