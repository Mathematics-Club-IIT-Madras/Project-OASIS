library(ISLR)
attach(Auto)
cylinders = as.factor(cylinders)
plot(cylinders, mpg)

plot(cylinders, mpg, col="red")

plot(cylinders, mpg, col="red", varwidth=T)

plot(cylinders, mpg, col="red", varwidth=T, horizontal=T)

plot(cylinders, mpg, col="red", varwidth=T, xlab="cylinders", ylab="MPG")