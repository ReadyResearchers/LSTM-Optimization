library(forecast)
library(clock)

data <- read.csv("../data/AEP_hourly.csv", header=TRUE)
data <- slice(data, 1:10000)
data <- mutate(data, Datetime = date_time_parse(Datetime, format = "%Y-%m-%d %H:%M:%S", zone = "UTC"))
data <- mutate(data, year = get_year(Datetime), month = get_month(Datetime), day = get_day(Datetime),
               hour = get_hour(Datetime))
data <- data[order(data$Datetime), ]

initial <- data$Datetime[1]

ts_data <- ts(data$AEP_MW, frequency = 24,
              start = c(get_year(initial), get_month(initial), get_day(initial),
                        get_hour(initial)))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Energy Consumption Data from AEP")


decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
