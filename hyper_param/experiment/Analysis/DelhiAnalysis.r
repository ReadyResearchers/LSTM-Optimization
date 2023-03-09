library(forecast)
library(clock)

data <- read.csv("../data/DailyDelhiClimateTrain.csv", header=TRUE)
data <- mutate(data, date = date_parse(date, format = "%Y-%m-%d"))
data <- mutate(data, year = get_year(date), month = get_month(date), day = get_day(date))

ts_data <- ts(data$meantemp, frequency = 365, start = c(data$year, data$month, data$day))

acf(ts_data, lag.max = 720, main = "Autocorrelation Plot of Temperature Data from Delhi")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
