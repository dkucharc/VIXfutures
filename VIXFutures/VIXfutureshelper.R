# function: restrict_to_start_end_date Select a subset of the data, based on a given date range and a given time range input: data: dataframe
# with the columns 'ENTRY_TIME' and 'TRADE_DATE' or 'TIME_STAMP' and 'TRADE_DT' data_type: string, either 'tick' or 'spot'. Choose 'tick' if
# your dataframe has the columns 'ENTRY_TIME' and 'TRADE_DATE', 'spot' if it has the columns 'TIME_STAMP' and 'TRADE_DT'

# output: data frame restricted to the given date and time range

restrict_to_start_end_date = function(data, data_type = "tick", start_time = "02:15:00", end_time = "15:15:00", start_date = "2018-01-02", end_date = "2018-02-22") {
    
    # VIX Tick data
    if (data_type == "tick") {
        # Time range
        data = data[which(data$ENTRY_TIME >= start_time & data$ENTRY_TIME <= end_time), ]
        # Date range
        data = data[which(data$TRADE_DATE >= start_date & data$TRADE_DATE <= end_date), ]
        
        # VIX Spot data
    } else if (data_type == "spot") {
        # Time range
        data = data[which(data$TIME_STAMP >= start_time & data$TIME_STAMP <= end_time), ]
        # Date range
        data = data[which(data$TRADE_DT >= start_date & data$TRADE_DT <= end_date), ]
    }
    return(data)
}
# the raw data has a time stamp where we want to change the formatting 02JAN18 will be the character 2018-01-02 The input, a dataframe, needs
# to have the columns 'ENTRY_TIME' and 'TRADE_DATE' The output is a dataframe where the formatting of the dates in the above two columns has
# changed Example: - Factor 02JAN18 will be character 2018-01-02 - Factor ENTRY_TIME will be times ENTRY_TIME VIXFutures =
# edit_time_date(VIXFutures = VIXFutures)
edit_time_date = function(VIXFutures) {
    
    # ENTRY_TIME
    VIXFutures$ENTRY_TIME = chron(times. = VIXFutures$ENTRY_TIME)
    
    # TRADE_DATE
    trade_date = matrix(VIXFutures$TRADE_DATE, ncol = 1)
    
    # convert date formatting to from 02JAN18 to 2018-01-02
    dates = apply(trade_date, MARGIN = 1, FUN = function(x) {
        substr_date = c(substr(x, 1, 2), substr(x, 3, 5), substr(x, 6, 9))
        x = paste(substr_date[1], substr_date[2], substr_date[3], sep = "-")
        x = strptime(x, format = "%d-%b-%Y")
        x = format(as.Date(x), "20%y-%m-%d")
        return(x)
    })
    
    VIXFutures$TRADE_DATE = dates
    
    return(VIXFutures)
}

## create data minute-by-minute resolution, both for the VIX index and the VIX future
create_minute_data = function(data, data_type = "tick", feature = "MID_QUOTE") {
    if (data_type == "tick") {
        # new column with datetime
        VIXFutures = data %>% mutate(Datetime = as.POSIXct(paste(TRADE_DATE, ENTRY_TIME, sep = " ")))
        # minutes are rounded up
        VIXFutures$Datetime = ceiling_date(VIXFutures$Datetime, "minute")
        VIXFutures$relevant_data = VIXFutures[, feature]
        # select the relevant data
        VIXFutures = VIXFutures %>% select(Datetime, relevant_data)
        # new data frame with the relevant data
        VIXFutures_rd = as.data.frame(unique(VIXFutures$Datetime))
        VIXFutures_rd$Index = VIXFutures_rd$`unique(VIXFutures$Datetime)`
        VIXFutures_rd$relevant_data = NA
        VIXFutures_rd = VIXFutures_rd %>% select(Index, relevant_data)
        # for loop which returns the last observation of a minute
        for (i in 1:length(VIXFutures_rd$relevant_data)) {
            VIXFutures_rd$relevant_data[i] = tail(VIXFutures$relevant_data[VIXFutures$Datetime == VIXFutures_rd$Index[i]], n = 1)
        }
        
        # column names change:
        colnames(VIXFutures_rd) = c("Datetime", "MID_QUOTE")
        return(VIXFutures_rd)
        
    } else if (data_type == "spot") {
        # new column with datetime
        VIXSpot = data %>% mutate(Datetime = as.POSIXct(paste(TRADE_DT, TIME_STAMP, sep = " ")))
        # minutes are rounded up
        VIXSpot$Datetime = ceiling_date(VIXSpot$Datetime, "minute")
        # select the relevant data
        VIXSpot = VIXSpot %>% select(Datetime, INDEX_VALUE)
        # new data frame with the relevant data
        VIXSpot_rd = as.data.frame(unique(VIXSpot$Datetime))
        VIXSpot_rd$Index = VIXSpot_rd$`unique(VIXSpot$Datetime)`
        VIXSpot_rd$relevant_data = NA
        VIXSpot_rd = VIXSpot_rd %>% select(Index, relevant_data)
        # for loop which returns the last observation of a minute
        for (i in 1:length(VIXSpot_rd$relevant_data)) {
            VIXSpot_rd$relevant_data[i] = tail(VIXSpot$INDEX_VALUE[VIXSpot$Datetime == VIXSpot_rd$Index[i]], n = 1)
        }
        # column names change:
        colnames(VIXSpot_rd) = c("Datetime", "INDEX_VALUE")
        return(VIXSpot_rd)
    }
}
