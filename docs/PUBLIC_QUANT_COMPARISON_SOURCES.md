# Public Quant Model Comparison - Sources & Method Notes

This document provides source references for the `README` comparison section and
`docs/public_quant_model_comparison.csv`.

## Source keys

### `lch_forbes_2024`

- Forbes summary of LCH Investments 2024 ranking, including cumulative net gains
  (Citadel, D.E. Shaw, Millennium) and annual return context:
  - https://www.forbes.com/sites/hanktucker/2025/01/19/citadel-de-shaw-and-the-worlds-top-20-hedge-funds-gained-a-record-94-billion-in-2024/
- Reuters coverage of LCH ranking framework and top-manager performance:
  - https://www.reuters.com/markets/hedge-funds-deliver-double-digit-returns-2024-2025-01-02/
  - https://www.reuters.com/business/finance/hedge-funds-have-charged-almost-2-trillion-fees-since-1969-says-lch-2025-01-20/

### `reuters_deshaw_2026_lch`

- Reuters report citing D.E. Shaw Composite return context and lifetime gain
  ranking references:
  - https://www.reuters.com/business/finance/de-shaws-flagship-funds-trump-market-volatility-beat-sp-500-2025-source-says-2026-01-02/

### `pionline_bridgewater_2015`

- Pensions & Investments report on Pure Alpha long-term annualized return
  disclosure:
  - https://www.pionline.com/article/20150219/INTERACTIVE/150219857/bridgewater-s-pure-alpha-strategy-up-more-than-13-annually-since-its-inception
- Reuters historical performance snapshots:
  - https://www.reuters.com/article/idUSL1N13J1XX20151124/
  - https://www.reuters.com/article/hedgefunds-bridgewater/bridgewaters-pure-alpha-ends-2018-with-146-pct-gain-source-idUSL1N1Z609N/

### `medallion_book_cornell`

- Gregory Zuckerman, *The Man Who Solved the Market* (publicly cited Medallion
  gross/net return history).
- Bradford Cornell analysis note / summary using disclosed historical numbers:
  - https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html

### `ishares_mtum_factsheet_2024`

- Official iShares MTUM fact sheet (AUM, historical annualized returns):
  - https://www.ishares.com/us/literature/fact-sheet/mtum-ishares-msci-usa-momentum-factor-etf-fund-fact-sheet-en-us.pdf

### `aqr_managed_futures_2025`

- Official AQR Managed Futures Strategy Fund page (AUM and annualized returns):
  - https://funds.aqr.com/funds/aqr-managed-futures-strategy-fund

### `man_ahl_shareclass_public`

- Public factsheet / public fund profile for Man AHL Diversified share-class
  statistics:
  - https://www.man.com/document?doc-type=TAP&instrument=rs30508&locale=en
  - https://www.trustnet.com/factsheets/B/ED50/man-ahl-diversified-plc-dn

### `this_repo_test_snapshot`

- Repository-generated test snapshot in README from:
  - `outputs/equity_curve.csv`
  - `outputs/model_result_summary.json`

## Method notes (important)

1. **Not apples-to-apples**:
   the sample mixes private hedge funds, public ETFs, mutual funds, and a demo
   research strategy.
2. **Time windows differ**:
   some returns are since inception, some are 10Y trailing, and the repo result
   is a short test window.
3. **AUM/capacity and returns are dynamic**:
   values may change materially with flows, lockups, leverage, and market regime.
4. **Estimated annual dollar profit**:
   `AUM_or_capacity * annualized_return` is used only for rough scale intuition,
   not realized or guaranteed investor PnL.
5. **Educational use only**:
   this section is for benchmarking context, not investment advice.
