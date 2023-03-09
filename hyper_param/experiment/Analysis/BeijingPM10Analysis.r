library(forecast)
library(dplyr)

data <- read.csv("../data/guanyuan_air_quality_data_original.csv", header=TRUE)
data <- na.omit(data)
data <- slice(data, 1:10000)

ts_data <- ts(data$PM10, frequency = 24,
              start = c(data$year[1], data$month[1], data$day[1]))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Beijing Air Quality Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
