library(forecast)
library(clock)
library(dplyr)

data <- read.csv("../data/jena_climate_data_original.csv", header = TRUE)
data <- slice(data, 1:10000)
data <- mutate(data, Date.Time = date_time_parse(Date.Time, format = "%d.%m.%Y %H:%M:%S", zone = "UTC"))
data <- filter(data, get_minute(data$Date.Time) == 0)
data <- mutate(data, Year = get_year(Date.Time), Month = get_month(Date.Time),
               Day = get_day(Date.Time), Hour = get_hour(Date.Time))

initial <- data$Date.Time[1]

ts_data <- ts(data$T..degC., frequency = 24,
              start = c(get_year(initial), get_month(initial), get_day(initial), get_hour(initial)))

acf(ts_data, lag.max = 168, main = "Autocorrelation Plot of Climate Data from Jena, Germany")

decomp <- decompose(ts_data)
covariance <- cov(decomp$seasonal, ts_data)
