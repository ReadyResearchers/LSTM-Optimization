library(forecast)
library(dplyr)

data <- read.csv("../data/norway_new_car_sales_by_make.csv", header=TRUE)
toyota_data <- filter(data, Make == "Toyota")

ts_data <- ts(toyota_data$Quantity, frequency = 12, start = c(toyota_data$Year, toyota_data$Month))

acf(ts_data, lag.max=60, main = "Autocorrelation Plot of Toyota Sales Data")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
