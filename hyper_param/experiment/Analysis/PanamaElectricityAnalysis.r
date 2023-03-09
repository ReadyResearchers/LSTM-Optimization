library(forecast)
library(clock)
library(dplyr)

data <- read.csv("../data/electricity_panama.csv", header=TRUE)
data <- na.omit(data)
data <- slice(data, 1:10000)
data <- mutate(data, datetime = date_time_parse(datetime, format = "%d-%m-%Y %H:%M", zone = "UTC"))
data <- mutate(data, year = get_year(datetime), month = get_month(datetime),
               day = get_day(datetime), hour = get_hour(datetime))

initial <- data$datetime[1]

ts_data <- ts(data$nat_demand, frequency = 24,
              start = c(get_year(initial), get_month(initial), get_day(initial)))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Panama Electricity Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
