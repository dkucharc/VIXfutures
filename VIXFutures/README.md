[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **VIXfutures** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet : VIXfutures

Published in : 'Neural Networks and Arbitrage in the VIX – A Deep Learning Approach for the VIX'

Description : 'Contains the VIX index and the VIX future intraday analysis on the tail-event day of February 5, 2018. 
The next day, Credit Suisse lost 500m USD because of a structured product on the VIX. 
Market disruption can already be observed the day before.'

Keywords : 
 - VIX
 - futures
 - arbitrage
 - market manipulation
 - index

See also : 

Author : Joerg Osterrieder

Submitted :  Fri, Feb 8 2019 by Joerg Osterrieder

Datafile : VIXData.RData, VIXfutureshelper.R

Example: VIX index and VIX futures data.
```

![Picture1](VIXIndex_Future.png)

### R Code
```r

rm(list = ls(all = TRUE))
# install and load packages
libraries = c("plotly", "formatR", "quantmod", "chron", "dplyr", "lubridate", "xts", "plyr", "car", "nortest", "tseries", "urca", "RColorBrewer", 
              "dgof", "ks", "KernSmooth", "zoo")
lapply(libraries, function(x) if (!(x %in% rownames(installed.packages()))) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)
source("VIXfutureshelper.R")
# please change your working directory setwd('C:/...')
# load the data, vix futures and vix index prices, intraday
load("data/VIXQuoteTickDataFrame.rda")
head(vix_quote_tick_df)

load("data/VIXSpotDataFrame.rda")
head(vix_spot_df)
# Analyse the VIX market on the 5th of February, 2018. The day after, Credit Suisse reportedly lost 500m USD because of a structured product
# on the VIX The VIX futures and the VIX index have substantially deviated from each other already the day before, indicating substantial
# turmoil in the VIX markets.
VIXindex = restrict_to_start_end_date(vix_spot_df, data_type = "spot", start_time = "08:30:00", end_time = "15:15:00", start_date = "2018-02-05", 
                                      end_date = "2018-02-05")
head(VIXindex)
VIXfutures = restrict_to_start_end_date(vix_quote_tick_df, data_type = "tick", start_time = "08:30:00", end_time = "15:15:00", start_date = "2018-02-05", 
                                        end_date = "2018-02-05")
# Use the front month future
VIXfutures = VIXfutures %>% filter(EXPR_DATE == "FEB18")
head(VIXfutures)
# Align times for the VIX index (published every 15 seconds) and the VIX futures data (tick data) and use minute-by-minute data
VIXfutures_rd = create_minute_data(VIXfutures, data_type = "tick", feature = "MID_QUOTE")
VIXindex_rd = create_minute_data(VIXindex, data_type = "spot")

# merge the VIX future and the VIX index based on 'Datetime'
VIXfutures_Index = merge(VIXfutures_rd, VIXindex_rd, by = c("Datetime"), all = FALSE)
head(VIXfutures_Index)

# convert to xts for easier plotting
VIXfutures_Index = xts(VIXfutures_Index[, c("MID_QUOTE", "INDEX_VALUE")], order.by = VIXfutures_Index$Datetime)
colnames(VIXfutures_Index) = c("Futures", "VIX Index")
head(VIXfutures_Index)

# plot the index and the future
png(filename = "VIXIndex_Future.png", width = 924, height = 651)
colors = brewer.pal(n = 6, name = "Paired")[c(6, 2)]
plot(as.zoo(VIXfutures_Index), observation.based = T, legend.loc = "topleft", screens = 1, grid.ticks.lty = 3, major.ticks = "hours", ylab = "price", 
     xlab = "time", main = "VIX index and future on February 5, 2018", grid.ticks.on = "hours", yaxis.right = FALSE, col = colors, lty = 1:2,
     lwd = c(4, 4), yaxis.same = TRUE)
legend("topleft", legend = c("VIX future", "VIX index"), col = colors, lty = 1:2, lwd = c(4, 4))
dev.off()
```

automatically created on 2019-02-08
