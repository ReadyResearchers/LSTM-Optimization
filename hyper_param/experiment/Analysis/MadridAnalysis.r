library(forecast)
library(clock)
library(dplyr)

data <- read.csv("../data/madrid_weather_data_original.csv", header=TRUE)
data <- mutate(data, time = date_time_parse(time, format = "%Y-%m-%d %H:%M:%S", zone = "UTC"))
data <- filter(data, get_hour(time) == 0)
data <- mutate(data, year = get_year(time), month = get_month(time), day = get_day(time))

initial <- data$time[1]

ts_data <- ts(data$temperature, frequency = 365,
              start = c(get_year(initial), get_month(initial), get_day(initial)))

acf(ts_data, lag.max = 730, main = "Autocorrelation Plot of Madrid Weather Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
