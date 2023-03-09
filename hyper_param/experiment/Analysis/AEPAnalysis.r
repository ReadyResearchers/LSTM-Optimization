library(forecast)
library(lubridate)

data <- read.csv("../data/AEP_hourly.csv", header=TRUE)
initial <- data$Datetime[1]

ts_data <- ts(data$AEP_MW, frequency = 24,
              start = c(year(initial), month(initial), day(initial)))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Energy Consumption Data from AEP")


decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
