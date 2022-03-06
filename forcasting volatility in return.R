set.seed(12345)
rm(list = ls())
graphics.off()

library(fGarch)
library(RSNNS)
library(rnn)
library(neuralnet)



AMZN<- read.csv("AMZN.csv")

data = AMZN

Return = diff(log(data$Close))

n = length(Return)

# for one year forecasting we will consider the last 250 as the test set
n.test =250

## training set of the log return

Return.train = Return[1:(n-n.test)]

## subtract the mean of the training set

Return.mean = Return - mean(Return.train)

## training set demeaned on the Return

Return.train = Return.mean[1:(n-n.test)]

## test set demeaned on the return 

Return.test = Return.mean[(n-n.test + 1):n]

n.train = length(Return.train)

#### Calculating the conditional volatility

## Data preparation

Return.fct = c(Return.train[n.train], Return.test[1:(n.test-1)])

## Forecasting the volatility using the GARCH(1,1) model
# fit the Garch model

Return.garch11 = garchFit(~garch(1,1), Return.train, cond.dist = "std", trace = FALSE)
Return.garch11

# calculate the volatility

sigma.train = Return.garch11@sigma.t

sigma.train

## fit the APPARCH model with different coefficient 

A0 = as.numeric(Return.garch11@fit$coef[2])
A0
A1 = as.numeric(Return.garch11@fit$coef[3])
A1
B1 = as.numeric(Return.garch11@fit$coef[4])
B1
B2 = as.numeric(Return.garch11@fit$coef[5])
B2

## correlation between the squared returns and the conditional variance or the volatility

cor(Return.train**2, sigma.train**2)

# calculate the conditional volatility for the test set

sigma.test = sqrt(A0+A1*Return.fct[1]**2+B1*sigma.train[n.train]**2)

for(i in 2:n.test){
  sigma.i = sqrt(A0+A1*Return.fct[i]**2+B1*sigma.test[i-1]**2)
  sigma.test = c(sigma.test, sigma.i)
}

cor(Return.test**2, sigma.test**2)

#### We have forecasting the volatility in return with the Garch(1,1) model,
#### now we have to modeling using the NNs methods:
### Data preparation for the NN-AR(p) model with p = 2

p = 3

X = paste0("V", 2:(p+1))

frml = as.formula(paste("V1~", paste(X, collapse = "+")))

## create a matrix with n-p rows and 4 columns

x = matrix(0, (n-p), (p+1))

## scale the target variable to range [0,1]

x[,1] = (Return[(p+1):n]^2-min(Return[(p+1):n]^2))/(max(Return[(p+1):n]^2)-min(Return[(p+1):n]^2))

## fill in the input columns 

for(i in 1:p){
  x[,(i+1)] = (Return[(p+1-i):(n-i)]^2-min(Return[(p+1-i):(n-i)]^2))/(max(Return[(p+1-i):(n-i)]^2)-min(Return[(p+1-i):(n-i)]^2))
}

y = as.data.frame((x))
y.train = x[1:(n.train-p),] 
y.test = x[(n.train-p+1):(n-p),]

#### Modeling the conditional variance with the function neuralnet(), p = 3

hidden = 10

model = neuralnet(frml, y.train, hidden = hidden)

## plot the model

plot(model)

## evaluate the results of the training set

results.train = compute(model, y.train[,2:(p+1)])

pred.train = results.train$net.result

cor.train = cor(pred.train, y.train[,1])
cor.train

## evaluate the result of the test set

results.test = compute(model, y.test[,2:(p+1)])

pred.test = results.test$net.result

cor.test = cor(pred.test, y.test[,1])
cor.test

##### plot the results from the Garch and the NN-ARCH models

model.com = mean(Return[(p+1):n.train]^2/pred.train)

par(mfcol = c(4,2))
par(mar = c(2,2,2,2))
plot.ts(abs(Return.test))
title("Test absolute returns, AMZN, 17.06.2018")

plot.ts(sigma.test)
title("Rolling volatility forecasts by GARCH")

plot.ts(Return.test/sigma.test)
title("Standardized test returns by GARCH")

acf((Return.test/sigma.test)^2, main="")
title("acf or sq. st. test return by GARCH")

plot.ts((y.train[,1]*model.com)**0.5)
title("Recalled absolute test returns")

plot.ts((pred.test*model.com)**0.5)
title("Rolling volatility forecasts by NN-ARCH")

plot.ts(Return.test/(pred.test*model.com)**0.5)
title("standardized test returns by NN-ARCH")

acf(Return.test^2/(pred.test*model.com), main="")
title("acf of sq. st. test retruns by NN-ARCH")

### modeling the volatility with the elman() function p = 4
## preparing the data for NN-AR(p)
p = 5

x = matrix(0, (n-p), p)
y = (Return[(p+1):n]^2-min(Return[(p+1):n]^2))/(max(Return[(p+1):n]^2)-min(Return[(p+1):n]^2))

for(i in 1:p){
  x[,i] = (Return[(p+1-i):(n-i)]^2-min(Return[(p+1-i):(n-i)]^2))/(max(Return[(p+1-i):(n-i)]^2)-min(Return[(p+1-i):(n-i)]^2))
}

x.train = x[1:(n.train-p),]
y.train = y[1:(n.train-p)]
x.test = x[(n.train-p+1):(n-p),]
y.test = x[(n.train-p+1):(n-p)]

### modeling with RNNS 

model_1 = elman(x.train, y.train, size = c(3,1), learnFuncParams = c(0.1), maxit = 1000,
                inputsTest = x.test, targetsTest = y.test, linOut = FALSE)


pred_1.train = predict(model_1, x.train)
pred_1.test = predict(model_1, x.test)

cor(y.train, pred_1.train)
cor(y.test, pred_1.test)

c.re = mean(Return[(p+1):n.train]^2/pred_1.train)

par(mfcol = c(4,2))
par(mar = c(2,2,2,2))
plot.ts(abs(Return.test))
title("Test absolute returns, AMZN, 17.06.2018")

plot.ts(sigma.test)
title("Rolling volatility forecasts by GARCH")

plot.ts(Return.test/sigma.test)
title("Standardized test returns by GARCH")

acf((Return.test/sigma.test)^2, main="")
title("acf or sq. st. test return by GARCH")

plot.ts((y.test*c.re)**0.5)
title("Recalled absolute test returns")

plot.ts((pred_1.test*c.re)**0.5)
title("Rolling volatility forecasts by NN-ARCH")

plot.ts(Return.test/(pred_1.test*c.re)**0.5)
title("standardized test returns by NN-ARCH")

acf(Return.test^2/(pred_1.test*c.re), main="")
title("acf of sq. st. test retruns by NN-ARCH")

### Modeling the volatility with the jordan() function with p = 2
## preparing the data

p = 6
x = matrix(0, (n-p), p)
y = (Return[(p+1):n]^2-min(Return[(p+1):n]^2))/(max(Return[(p+1):n]^2)-min(Return[(p+1):n]^2))

for(i in 1:p){
  x[,i] = (Return[(p+1-i):(n-i)]^2-min(Return[(p+1-i):(n-i)]^2))/(max(Return[(p+1-i):(n-i)]^2)-min(Return[(p+1-i):(n-i)]^2))
}

x.train = x[1:(n.train-p),]
y.train = y[1:(n.train-p)]
x.test = x[(n.train-p+1):(n-p),]
y.test = x[(n.train-p+1):(n-p)]

model_2 = jordan(x.train, y.train, size = c(1), maxit = 100, initFunc = "JE_Weights",
                 initFuncParams = c(1, -1, 0.3, 1, 0.5), learnFunc = "JE_BP", learnFuncParams = c(0.2),
                 shufflePatterns = FALSE, linOut = TRUE, inputsTest = NULL, targetsTest = NULL)
pred_2.train = predict(model_2, x.train)
pred_2.test = predict(model_2, x.test)

cor(y.train, pred_2.train)
cor(y.test, pred_2.test)

c.re = mean(Return[(p+1):n.train]^2/pred_2.train)

par(mfcol = c(4,2))
par(mar = c(2,2,2,2))
plot.ts(abs(Return.test))
title("Test absolute returns, AMZN, 17.06.2018")

plot.ts(sigma.test)
title("Rolling volatility forecasts by GARCH")

plot.ts(Return.test/sigma.test)
title("Standardized test returns by GARCH")

acf((Return.test/sigma.test)^2, main="")
title("acf or sq. st. test return by GARCH")

plot.ts((y.test*c.re)**0.5)
title("Recalled absolute test returns")

plot.ts((pred_2.test*c.re)**0.5)
title("Rolling volatility forecasts by NN-ARCH")

plot.ts(Return.test/(pred_2.test*c.re)**0.5)
title("standardized test returns by NN-ARCH")

plot(pred_2.test, y.test, pch = "+")
title("test sample predited volatility & sq-r")

### modeling the volatility with the mlp() function and p = 10

p = 10
x = matrix(0, (n-p), p)
y = (Return[(p+1):n]^2-min(Return[(p+1):n]^2))/(max(Return[(p+1):n]^2)-min(Return[(p+1):n]^2))

for(i in 1:p){
  x[,i] = (Return[(p+1-i):(n-i)]^2-min(Return[(p+1-i):(n-i)]^2))/(max(Return[(p+1-i):(n-i)]^2)-min(Return[(p+1-i):(n-i)]^2))
}

x.train = x[1:(n.train-p),]
y.train = y[1:(n.train-p)]
x.test = x[(n.train-p+1):(n-p),]
y.test = x[(n.train-p+1):(n-p)]

model_3 = mlp(x.train, y.train, size = c(10, 10), learnFuncParams = c(0.1), maxit = 1000, linOut = FALSE)

pred_3.train = predict(model_3, x.train)
pred_3.test = predict(model_3, x.test)

cor(y.train, pred_3.train)
cor(y.test, pred_3.test)

c.re = mean(Return[(p+1):n.train]^2/pred_3.train)

par(mfcol = c(4,2))
par(mar = c(2,2,2,2))
plot.ts(abs(Return.test))
title("Test absolute returns, AMZN, 17.06.2018")

plot.ts(sigma.test)
title("Rolling volatility forecasts by GARCH")

plot.ts(Return.test/sigma.test)
title("Standardized test returns by GARCH")

acf((Return.test/sigma.test)^2, main="")
title("acf or sq. st. test return by GARCH")

plot.ts((y.test*c.re)**0.5)
title("Recalled absolute test returns")

plot.ts((pred_3.test*c.re)**0.5)
title("Rolling volatility forecasts by NN-ARCH")

plot.ts(Return.test/(pred_3.test*c.re)**0.5)
title("standardized test returns by NN-ARCH")

acf(Return.test^2/(pred_3.test*c.re), main="")
title("acf of sq. st. test retruns by NN-ARCH")

