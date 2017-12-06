#############################################
# Wine Quality
# December, 2017
# Arthor: Yina Wei  Uni: yw2922
#############################################


rm(list=ls())
library(RSNNS)
library(ggplot2)
library(nnet)
library(dplyr)
library(glmnet)
library(randomForest)
library(corrplot)
setwd("/Users/aple/Desktop/ads/project 5")

# Data Reading

data.red = read.csv("winequality-red.csv", sep=";")
data.white = read.csv("winequality-white.csv", sep=";")
data.red$color = 1
data.white$color = 0
data = rbind(data.red, data.white)

# Explanatory Data Analysis

## Histogram
plotdata = data
plotdata[plotdata$color==1,"color"] = "red"
plotdata[plotdata$color==0,"color"] = "white"
hist(data$quality)
qplot(quality, 
      data = plotdata, 
      fill = color, 
      binwidth = 1, 
      origin = - 0.5, 
      main = "Quality of Red and White Wine") +
  scale_x_continuous(breaks = seq(2,10,1), lim = c(2,10)) +
  scale_y_sqrt(breaks = seq(0,5600,500)) +
  xlab('Quality') +
  ylab('Quantity')



# Correlation
corrplot.mixed(cor(data))
cor(x=data[,c(1:11,13)], y=data[,12])
# Alcohol (+++)
# Volatile acidity (---)
# Citric acid (++)
# Fixed acidity (+)
# Sulphates ?? worht investigating (+)
# Total sulphur dioxide (-)
# Density (-)
# Chlorides (-)






# PREDICTION
 
my.data = as.data.frame(scale(data[c(1:11,13)]))     # read and scale data
my.data$quality = data$quality
set.seed(1)
n = dim(my.data)[1]
train = sample(1:n,4330)   # 2/3 as training data, 1/3 as test data
train.data =  my.data[train,]
test.data = my.data[-train,]


# 1. Linear Regression ---------------------------------------------

# Lasso
x=as.matrix(train.data[,-13])
y=as.vector(train.data[,13])
#lambdas <- exp( seq(-3, 10, length=50))
# begin cv
set.seed(123)
cv.out <- cv.glmnet(x=x, y=y, nfolds=5, alpha=1, 
                 family='gaussian', intercept=TRUE)
plot(cv.out)
lambda_final = cv.out$lambda.min
# use final lambda to fit a LASSO model
fit.la <- glmnet(x=x, y=y, lambda=lambda_final,
                 family='gaussian', alpha=1, intercept=TRUE)
# test
predict.ls = predict(fit.la, as.matrix(test.data[,c(1:12)]))
plot(test.data[1:100,13])
points(predict.ls[1:100], col = "orange", pch = 2)
mse.ls = mean((predict.ls - test.data[,13])^2)
mse.ls
#0.52

# 2. 1NN -----------------------------------------------------------

net1 = mlp(x=train.data[,1:12], y=train.data[,13], size=c(2),    # 1 layer, 2 neuron
           maxit = 10000, learnFuncParams = c(0.1,0), linOut = T)  # fit neural network
predict.1nn = predict(net1, test.data[,c(1:12)])    # predict test set result
points(predict.1nn[1:100], col = "red", pch = 3)
mse.1nn = mean((predict.1nn - test.data[,13])^2)
mse.1nn
# 0.53

# 3. Ensemble NN -------------------------------------------------------------

net = list()
sizes = list(c(2), c(3), c(2,2), c(3,3), c(2,2,2), c(3,3,3),c(4,4,4),c(2,2,2,2),c(3,3,3,3),c(4,4,4,4))
a=1
for (i in 1:length(sizes)){
  for (j in 1:5){
    print(a)
    net[[a]] = mlp(x=train.data[,1:12], y=train.data[,13], size=sizes[[i]], 
                  maxit = 10000, linOut = T)
    a = a+1
  }
}
predict.ens = apply(sapply(net, function(x) predict(x, test.data[,1:12])), 1, mean)
points(predict.ens[1:100], col = "green", pch = 3)
mse.ens = mean((predict.ens - test.data[,13])^2)
mse.ens
# 0.49

# 4. Random Forest----------------------------------------------
rf = randomForest(quality~., train.data, mtry =4, importance = T)
par(mfrow = c(1,1))
varImpPlot(rf)
predict.rf = predict(rf, test.data[,1:12])
points(predict.rf[1:100], col = "blue", pch = 5)
mse.rf = mean((predict.rf - test.data[,13])^2)
#0.34