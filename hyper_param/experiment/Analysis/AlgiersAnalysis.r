library(forecast)
library(clock)

data <- read.csv("../data/city_temperature_compressed.csv", header=TRUE)

ts_data <- ts(data$AvgTemperature, frequency = 365, start = c(data$Year, data$Month, data$Day))

acf(ts_data, lag.max = 365, main = "Autocorrelation Plot of Temperature Data from Algiers")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
