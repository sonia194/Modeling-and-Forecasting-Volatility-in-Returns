set.seed(12345)
rm(list = ls())
graphics.off()

library(RSNNS)
library(rnn)
library(neuralnet)
library(fGarch)

data = read.csv("IBM.DE.csv")

return = diff(log(data$Close))

n = length(return)

# for one year forecasting we will consider the last 250 as the test set
n.test =250

## training sample of the log return

return.train = return[1:(n-n.test)]

## subtract the mean of the training sample

return.mean = return - mean(return.train)

## training sample demeaned on the Return

return.train = return.mean[1:(n-n.test)]

## test sample demeaned on the return 

return.test = return.mean[(n-n.test + 1):n]

n.train = length(return.train)

#### Calculating the conditional variance of the returns

## Data preparation

return.fct = c(return.train[n.train], return.test[1:(n.test-1)])

## Forecasting the volatility using the GARCH(1,1) model
# fit the Garch model

return.garch11 = garchFit(~garch(1,1), return.train, cond.dist = "std", trace = FALSE)

return.garch11

# calculate the volatility

sigma.train = return.garch11@sigma.t

## fit the APPARCH model with different coefficients

a0 = as.numeric(return.garch11@fit$coef[2])
a0
a1 = as.numeric(return.garch11@fit$coef[3])
a1
b1 = as.numeric(return.garch11@fit$coef[4])
b1
b2 = as.numeric(return.garch11@fit$coef[5])
b2

## correlation between the squared returns and the conditional variance or the volatility

cor(return.train**2, sigma.train**2)

# calculate the conditional volatility for the test set

sigma.test = sqrt(a0+a1*return.fct[1]**2+b1*sigma.train[n.train]**2)

for(i in 2:n.test){
  sigma.i = sqrt(a0+a1*return.fct[i]**2+b1*sigma.test[i-1]**2)
  sigma.test = c(sigma.test, sigma.i)
}

cor(return.test**2, sigma.test**2)

###############################################################################
## Build and train the FNN/ENN for frocasting the volatility with neuralnet()

## preparing the data for an NN-AR(p)

p = 10

# create the formula

Xnam = paste0("Y.L", 1:p)

frml = as.formula(paste("Yt~", paste(Xnam, collapse = "+")))

## create a matrix with n-p rows and 4 columns for the input variables

x = matrix(0, (n-p), (p+1))

## scale the target variable to range [0,1]

x[,1] = (return[(p+1):n]^2-min(return[(p+1):n]^2))/(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))

## fill in the input columns 

for(i in 1:p){
  x[,(i+1)] = (return[(p+1-i):(n-i)]^2-min(return[(p+1-i):(n-i)]^2))/(max(return[(p+1-i):(n-i)]^2)-min(return[(p+1-i):(n-i)]^2))
}

# define the variable's name in the matrix x

colnames(x)<-c("Yt", Xnam)

# scale the training sample

x.train = x[1:(n.train-p),]

# scale the test sample

x.test = x[(n.train-p+1):(n-p),]

# unscaled training sample

return.train = return[(p+1):(n-n.test)]

# unscaled  test sample

return.test = return[(n-n.test+1):n]

# fit a model on the training data with 12 hidden

M = 12

CorMat = matrix(0, 8, M)
for (i in 1:M) {set.seed(12345)
  model = neuralnet(frml, data = x.train, hidden = c(i), threshold = 0.02, stepmax = 1000000)
  
   # compute estimated training sample scaled return
  
  results.train  = compute(model, x.train[,2:(p+1)])
  
  pred.train = results.train$net.result
  
  # compute estimated test sample
  
  results.test = compute(model, x.test[,2:(p+1)])
  pred.test = results.test$net.result
  
  # rescale the estimated scaled volatility
  
  ret.pred.train = pred.train*(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  ret.pred.test = pred.test*(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
                             
  CorMat[1, i] = cor(x.train[,1], pred.train)
  CorMat[2, i] = cor(x.test[,1], pred.test)
  CorMat[3, i] = mean((return.train-ret.pred.train)^2)*10000
  CorMat[4, i] = mean((return.test-ret.pred.test)^2)*10000
  CorMat[5, i] = cor(x.train[,1], pred.train, method = "spearman")
  CorMat[6, i] = cor(x.test[,1], pred.test, method = "spearman")
  CorMat[7, i] = sum(return.train*ret.pred.train>0)/n.train 
  CorMat[8, i] = sum(return.test*ret.pred.test>0)/n.test  
}

## plot he model 
plot(model)

write.table(trunc(CorMat*1000+0.5)/1000, "IBM-returns-nn-1HL12.csv")
CorMat.nn12 = read.table("IBM-returns-nn-1HL12.csv")
CorMat = CorMat.nn12


# maximum test sample correlation

max(CorMat[2,])

which.max(CorMat[2,])

trunc(CorMat[2,]*1000+0.5)/1000

# minimum test MSE

min(CorMat[4,])

which.min(CorMat[4,])

trunc(CorMat[4,]*1000+0.5)/1000

#maximum test Spearman's rank correlation

max(CorMat[6,])

which.max(CorMat[6,])

trunc(CorMat[6,]*1000+0.5)/1000

#maximum test CPD

max(CorMat[8,])

which.max(CorMat[8,])

trunc(CorMat[8,]*1000+0.5)/1000

################################################################
## Build and train hidden layer FNN models with jordan()

p = 10
x = matrix(0, (n-p), p)

#define test variables

Y = (return[(p+1):n]^2-min(return[(p+1):n]^2))/(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))

## fill in the train columns 

for(i in 1:p){
  x[,i] = (return[(p+1-i):(n-i)]^2-min(return[(p+1-i):(n-i)]^2))/(max(return[(p+1-i):(n-i)]^2)-min(return[(p+1-i):(n-i)]^2))
}

# scale the training input variables

x.train = x[1:(n.train-p),]

# scale the training output

y.train =Y[1:(n.train-p)]

# scale the test input variables

x.test = x[(n.train-p+1):(n-p),]

y.test = Y[(n.train-p+1):(n-p)]

# unscaled training sample

return.train = return[(p+1):(n-n.test)]

# unscaled  test sample

return.test = return[(n-n.test+1):n]

# Fit the model on the training data
M = 12

CorMat=matrix(0, 8, M)

for(i in 1:M){set.seed(1234)
  model <- jordan(x.train, y.train, size = c(i), maxit = 250,
                  initFunc = "JE_Weights", initFuncParams = c(1, -1, 0.3, 1, 0.5),
                  learnFunc = "JE_BP", learnFuncParams = c(0.2),
                  updateFunc = "JE_Order", updateFuncParams = c(0),
                  shufflePatterns = FALSE, linOut = TRUE, inputsTest = NULL,
                  targetsTest = NULL)
  
  #compute estimated scaled returns
  pred.train <- predict(model, x.train)
  pred.test <- predict(model, x.test)
  
  ## transform the predicted values back
  ret.pred.train=pred.train*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  ret.pred.test=pred.test*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  
  CorMat[1, i]=cor(y.train, pred.train) 
  CorMat[2, i]=cor(y.test, pred.test) 
  CorMat[3, i]=mean((return.train-ret.pred.train)^2)*10000 
  CorMat[4, i]=mean((return.test-ret.pred.test)^2)*10000 
  CorMat[5, i]=cor(y.train, pred.train, method="spearman") 
  CorMat[6, i]=cor(y.test, pred.test, method="spearman")
  CorMat[7, i]=sum(return.train*ret.pred.train>0)/n.train 
  CorMat[8, i]=sum(return.test*ret.pred.train>0)/n.test 
}
write.table(trunc(CorMat*1000+0.5)/1000, "IBM-returns-jordan-1HL12.csv")
CorMat.jordan12=read.table("IBM-returns-jordan-1HL12.csv", header=TRUE)
CorMat=CorMat.jordan12

#maximum test correlation
max(CorMat[1,])

which.max(CorMat[1,])

trunc(CorMat[1,]*1000+0.5)/1000

#minimum test MSE
min(CorMat[3,])

which.min(CorMat[3,])

trunc(CorMat[3,]*1000+0.5)/1000

#maximum test Spearman's rank correlatoin
max(CorMat[5,])

which.max(CorMat[5,])

trunc(CorMat[5,]*1000+0.5)/1000

#maximum test CPD
max(CorMat[7,])

which.max(CorMat[7,])

trunc(CorMat[7,]*1000+0.5)/1000
##################################################################
## Build and train hidden layer FNN models with mlp()
p = 10

x=matrix(0, (n-p), p)

Y=(return[(p+1):n]^2-min(return[(p+1):n]^2))/(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))

for(i in 1:p){
  x[,i]=(return[(p+1-i):(n-i)]^2- min(return[(p+1-i):(n-i)]^2))/(max(return[(p+1-i):(n-i)]^2)-
                                                         min(return[(p+1-i):(n-i)]^2))
}
x.train=x[1:(n.train-p),] 
y.train=Y[1:(n.train-p)] 
return.train=return[(p+1):(n-n.test)]
x.test=x[(n.train-p+1):(n-p),] 
y.test=Y[(n.train-p+1):(n-p)] 
return.test=return[(n-n.test+1):n] 

#### Fit a model on the training data 
M=12
CorMat=matrix(0, 8, M)
for(i in 1:M){set.seed(1234)
  model <- mlp(x.train, y.train,
               size = c(i), learnFuncParams = c(0.1), maxit = 20000,
               linOut = FALSE)
  #compute the estimated scaled return
  pred.train <- predict(model, x.train)
  pred.test <- predict(model, x.test)
  ## transform the predicted values back
  ret.pred.train=pred.train*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  ret.pred.test=pred.test*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  CorMat[1, i]=cor(y.train, pred.train)

  CorMat[2, i]=cor(y.test, pred.test) 
  CorMat[3, i]=mean((return.train-ret.pred.train)^2)*10000 
  CorMat[4, i]=mean((return.test-ret.pred.test)^2)*10000 
  CorMat[5, i]=cor(y.train, pred.train, method="spearman") 
  CorMat[6, i]=cor(y.test, pred.test, method="spearman")
  CorMat[7, i]=sum(return.train*ret.pred.train>0)/n.train 
  CorMat[8, i]=sum(return.test*ret.pred.test>0)/n.test 
}


write.table(trunc(CorMat*1000+0.5)/1000, "IBM-returns-mlp-1HL12.csv")
CorMat.mlp12=read.table("IBM-returns-mlp-1HL12.csv")
CorMat=CorMat.mlp12

max(CorMat[1,])

which.max(CorMat[1,])

trunc(CorMat[1,]*1000+0.5)/1000

## minimum MSE_test and its location in the matrix
min(CorMat[3,])

which.min(CorMat[3,])

trunc(CorMat[3,]*1000+0.5)/1000

## maximum out-sample spearsman`s rank correlation and its location in the matrix
max(CorMat[5,])

which.max(CorMat[5,])

trunc(CorMat[5,]*1000+0.5)/1000

## maximum out-sample ratio CPD
max(CorMat[7,])

which.max(CorMat[7,])

trunc(CorMat[7,]*1000+0.5)/1000

####################################################################
## Build and train hidden layer FNN models with elman()

p = 10

x=matrix(0, (n-p), p)

Y=(return[(p+1):n]^2-min(return[(p+1):n]^2))/(max(return[(p+1):n]^2)-min(return[(p+1):n]^2))

for(i in 1:p){
  x[,i]=(return[(p+1-i):(n-i)]^2- min(return[(p+1-i):(n-i)]^2))/(max(return[(p+1-i):(n-i)]^2)-
                                                                   min(return[(p+1-i):(n-i)]^2))
}
x.train=x[1:(n.train-p),] 
y.train=Y[1:(n.train-p)] 
return.train=return[(p+1):(n-n.test)]
x.test=x[(n.train-p+1):(n-p),] 
y.test=Y[(n.train-p+1):(n-p)] 
return.test=return[(n-n.test+1):n] 

#### Fit a model on the training data 
M=12
CorMat=matrix(0, 8, M)
for(i in 1:M){set.seed(1234)
  model <- elman(x.train, y.train,
               size = c(i), learnFuncParams = c(0.18), maxit = 3700,
               linOut = FALSE)
  #compute the estimated scaled return
  pred.train <- predict(model, x.train)
  pred.test <- predict(model, x.test)
  ## transform the predicted values back
  ret.pred.train=pred.train*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  ret.pred.test=pred.test*(max(return[(p+1):n]^2)- min(return[(p+1):n]^2))+min(return[(p+1):n]^2)
  CorMat[1, i]=cor(y.train, pred.train)
  
  CorMat[2, i]=cor(y.test, pred.test) 
  CorMat[3, i]=mean((return.train-ret.pred.train)^2)*10000 
  CorMat[4, i]=mean((return.test-ret.pred.test)^2)*10000 
  CorMat[5, i]=cor(y.train, pred.train, method="spearman") 
  CorMat[6, i]=cor(y.test, pred.test, method="spearman")
  CorMat[7, i]=sum(return.train*ret.pred.train>0)/n.train 
  CorMat[8, i]=sum(return.test*ret.pred.test>0)/n.test 
}


write.table(trunc(CorMat*1000+0.5)/1000, "IBM-returns-elman-1HL12.csv")
CorMat.elman12=read.table("IBM-returns-elman-1HL12.csv")
CorMat=CorMat.elman12

max(CorMat[1,])

which.max(CorMat[1,])

trunc(CorMat[1,]*1000+0.5)/1000

## minimum MSE and its location in the matrix
min(CorMat[3,])

which.min(CorMat[3,])

trunc(CorMat[3,]*1000+0.5)/1000

## maximum  speasrman`s rank correlation and its location in the matrix
max(CorMat[5,])

which.max(CorMat[5,])

trunc(CorMat[5,]*1000+0.5)/1000

## maximum out-sample ratio CPD
max(CorMat[7,])

which.max(CorMat[7,])

trunc(CorMat[7,]*1000+0.5)/1000

#########################################################################

## Model performance visualization

#### figure 1 for elman and jordan with 1 HL
par(mfrow=c(2,2))
CorMat.elman12=read.table("IBM-returns-elman-1HL12.csv")
CorMat=CorMat.elman12
Node = 1:M
matplot(Node, t(CorMat[1:4,]), type="lblb", pch="orom",
        xlab="Node (vertical line for a double choice)",
        ylab="corr, mse w. r & m for Y.te",
        main="Correlation & MSE for elman with 1 HL", cex.main=1)

abline(v=8, col=2)
abline(v=8, col=4)

matplot(Node, t(CorMat[5:8,]), type="lblb", pch="opod",
        xlab="Node (vertical lines for the choices)",
        ylab="r.p, cpd w. p & d for Y.te")
title("Spearman's rank corr. & CPD for elman with 1 HL", cex.main=1)
abline(v=7, col=6)
abline(v=1, col=8)

CorMat.jordan12=read.table("IBM-returns-jordan-1HL12.csv")
CorMat=CorMat.jordan12

matplot(Node, t(CorMat[1:4,]), type="lblb", pch="orom",
        xlab="Node (vertical line for a double choice)",
        ylab="corr, mse w. r & m for Y.te",
        main="Correlation & MSE for jordan with 1 HL", cex.main=1)

abline(v=2, col=2)
abline(v=7, col=4)

matplot(Node, t(CorMat[5:8,]), type="lblb", pch="opod",
        xlab="Node (vertical line for a double choice)",
        ylab="r.p, cpd w. p & d for Y.te")
title("Spearman's rank corr. & CPD for jordan with 1 HL", cex.main=1)
abline(v=2, col=6)
abline(v=1, col=8)



#### figure 2 for mlp and nn with 1 HL
par(mfrow=c(2,2))

CorMat.mlp12=read.table("IBM-returns-mlp-1HL12.csv", header=TRUE)
CorMat=CorMat.mlp12

matplot(Node, t(CorMat[1:4,]), type="lblb", pch="orom",
        xlab="Node (vertical lines for the choices)",
        ylab="corr, mse w. r & m for Y.te",
        main="Correlation & MSE for mlp with 1 HL", cex.main=1)

abline(v=12, col=2)
abline(v=7, col=4)

matplot(Node, t(CorMat[5:8,]), type="lblb", pch="opod",
        xlab="Node (vertical line for a double choice)",
        ylab="r.p, cpd w. p & d for Y.te")

title("Spearman's rank corr. & CPD for mlp with 1 HL", cex.main=1)

abline(v=2, col=6)
abline(v=1, col=8)

CorMat.nn12=read.table("IBM-returns-nn-1HL12.csv", header=TRUE)
CorMat=CorMat.nn12

matplot(Node, t(CorMat[1:4,]), type="lblb", pch="orom",
        xlab="Node (vertical line for a double choice)",
        ylab="corr, mse w. r & m for Y.te",
        main="Correlation & MSE for nn with 1 HL", cex.main=1)

abline(v=3, col=2)
abline(v=12, col=4)

matplot(Node, t(CorMat[5:8,]), type="lblb", pch="opod",
        xlab="Node (vertical lines for the choices)",
        ylab="r.p, cpd w. p & d for Y.te")
title("Spearman's rank corr. & CPD for nn with 1 HL", cex.main=1)

abline(v=3, col=6)
abline(v=4, col=8)






