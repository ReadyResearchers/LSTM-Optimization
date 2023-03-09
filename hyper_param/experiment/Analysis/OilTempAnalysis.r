library(forecast)
library(lubridate)

data <- read.csv("../data/ETTh1.csv", header=TRUE)
initial <- data$date[1]

ts_data <- ts(data$OT, frequency = 24,
              start = c(year(initial), month(initial), day(initial)))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Oil Temperature Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
