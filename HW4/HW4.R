library(MASS)
library(class)
library(leaps)
library(boot)
library(randomForest)
library(gbm)
library(ISLR)
library(tree)

setwd("/Users/apple/Desktop/590/H4")
creditcard <- read.csv("creditcard.csv")
head(creditcard)

# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# Ref: https://www.kaggle.com/dalpozz/creditcardfraud

n <- nrow(creditcard) # there are n records

# Arrange this dataset in a random order
set.seed(1)
d <- 50000 # select 50000 samples
random.order <- sample(seq(1,n), d)
creditcardsub <- creditcard[random.order, ]

# Divide dataset into two group: Training data and test data
training <- creditcardsub[seq(1, d/2), ]
test <- creditcardsub[seq(d/2+1, d), ]


# 2--------------

# The quantitative variable I chose is Amount
training1 <- training[,-31]
test1 <- test[,-31]

# 1. Best subset selection + logistic regression
m <- regsubsets(Amount~., data = training1, nvmax=30) 
regs <- t(summary(m)$which)

cp <- summary(m)$cp
i <- which.min(cp)
plot(cp,type='b',col="blue",
     xlab="Number of Predictors",ylab=expression("Mallows C"[P]))
points(i,cp[i],pch=19,col="red")

regs[, 18]

training2 <- training1[, -c(1, 12, 13, 14, 16, 17, 18, 25, 27, 28, 29)]
test2 <- test1[, -c(1, 12, 13, 14, 16, 17, 18, 25, 27, 28, 29)]
glm.fit <- glm(Amount~., data = training2) 
glm.pred <- predict(glm.fit, test2, type ="response")
mean((test2$Amount - glm.pred)^2)

# 2. Linear Discriminant Analysis
lda.fit <- lda(Amount~., data=training2)
lda.fit

lda.pred <- predict(lda.fit, test2, type ="response")
lda.x <- lda.pred$x
mean((test2$Amount - lda.x)^2)

# 3. Decision tree
tree.credit <- tree(Amount~.,training2)
summary(tree.credit)

plot(tree.credit)
text(tree.credit ,pretty=0)

cv.credit=cv.tree(tree.credit)
plot(cv.credit$size ,cv.credit$dev ,type='b')
# Doesn't need to prune
tree.pred <- predict(tree.credit, newdata=test2)
mean((test2$Amount - tree.pred)^2)

# 4. Knn
amount <- training1$Amount
train.knn <- as.matrix(training1[, -30])
test.knn <- as.matrix(test1[, -30])

knn.pred1 <- knn(train.knn, test.knn, amount, k = 1)
knn.pred2 <- knn(train.knn, test.knn, amount, k = 2)
knn.pred3 <- knn(train.knn, test.knn, amount, k = 3)
knn.pred1 <- as.numeric(as.vector(knn.pred1))
knn.pred2 <- as.numeric(as.vector(knn.pred2))
knn.pred3 <- as.numeric(as.vector(knn.pred3))
mean((test2$Amount - knn.pred1)^2)
mean((test2$Amount - knn.pred2)^2)
mean((test2$Amount - knn.pred3)^2)


# 3--------------

# The qualitative variable I chose is Class
Testclass <- table(test$Class)
Testclass

# Baseline accuracy
Testclass[1]/sum(Testclass)

# 1. Logistic Regression
glm.fit <- glm(Class~., data=training, family="binomial")
summary(glm.fit)

glm.probs <- predict(glm.fit, test, type ="response")

glm.pred <- rep(0, dim(test)[1])
glm.pred[glm.probs >.5] <- 1
table(test$Class, glm.pred)
mean(test$Class == glm.pred)
# 0.9992627

# 2. Linear Discriminant Analysis
lda.fit <- lda(Class~., data=training)
lda.fit
plot(lda.fit)

lda.pred <- predict(lda.fit, test)
lda.class <- lda.pred$class
table(test$Class, lda.class)
mean(test$Class == lda.class)
# 0.9993891

# 3. Quadratic Discriminant Analysis
qda.fit <- qda(Class~., data=training)
qda.fit

qda.pred <- predict(qda.fit, test)
qda.class <- qda.pred$class
table(test$Class, qda.class)
mean(test$Class == qda.class)
# 

# 4. K-Nearest Neighbors
class <- training$Class
train.knn <- as.matrix(training[, seq(1,30)])
test.knn <- as.matrix(test[, seq(1,30)])

knn.pred1 <- knn(train.knn, test.knn, class, k = 1)
knn.pred2 <- knn(train.knn, test.knn, class, k = 2)
knn.pred3 <- knn(train.knn, test.knn, class, k = 3)
table(test$Class, knn.pred1)
mean(test$Class == knn.pred1)
table(test$Class, knn.pred2)
mean(test$Class == knn.pred2)
table(test$Class, knn.pred3)
mean(test$Class == knn.pred3)


