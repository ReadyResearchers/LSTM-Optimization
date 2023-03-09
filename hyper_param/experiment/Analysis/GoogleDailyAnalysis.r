library(forecast)
library(clock)

data <- read.csv("../data/GOOG_daily_original.csv", header=TRUE)
data <- mutate(data, Date = date_parse(Date, format = "%Y-%m-%d"))
data <- mutate(data, year = get_year(Date), month = get_month(Date), day = get_day(Date))

initial <- data$Date[1]

ts_data <- ts(data$Close, frequency = 365,
              start = c(get_year(initial), get_month(initial), get_day(initial)))

acf(ts_data, lag.max = 730, main = "Autocorrelation Plot of Daily Google Stock Price Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
