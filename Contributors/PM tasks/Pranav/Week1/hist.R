library(ISLR)
attach(Auto)
cylinders = as.factor(cylinders)
hist(mpg)
hist(mpg, col=2)
hist(mpg, col=2, breaks=15)

pairs(Auto)
pairs(~ mpg + displacement + horsepower + weight + acceleration, Auto)