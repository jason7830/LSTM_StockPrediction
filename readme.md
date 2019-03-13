# LSTM_StockPrediction


> Training Stock historical data and eventually predict future prices.

> Ojective: Predict future 20-days price.

## To be solved
> It can only predict tommorow's price now with the error up to 30 percent. If so, the error must be uder 10 percent, because stocks in Taiwan have rise and drop limit set by the Financial Supervisory Commission. Or the result is considered has no reference value.

- Rebuild Model
- 每個檔案中有數列，每列為一天交易的資訊
- 每列包含：交易日期、成交股數、成交金額、開盤價、最高價、最低價、收盤價、漲跌價差、成交筆數，共 9 欄。
- 符號說明: +表示漲、- 表示跌、X表示不比價
- 當日統計資訊含一般、零股、盤後定價、鉅額交易，不含拍賣、標購。
