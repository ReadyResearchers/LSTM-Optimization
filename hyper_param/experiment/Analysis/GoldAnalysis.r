library(forecast)
library(clock)

data <- read.csv("../data/gold_price_data_original.csv", header = TRUE)
data <- slice(data, 1:10000)
data <- mutate(data, Date = date_parse(Date, format = "%Y-%m-%d"))
data <- mutate(data, Year = get_year(Date), Month = get_month(Date), Day = get_day(Date))


ts_data <- ts(data$Value, frequency = 365,
              start = c(data$year, data$Month, data$Day, data$Hour))

acf(ts_data, lag.max = 365, main = "Autocorrelation Plot of Gold Price Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
