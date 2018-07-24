
library(MASS)
library(boot)
library(randomForest)
library(gbm)

index = sample(1:nrow(Boston), nrow(Boston)/2)
# bag tree function
bag_function <- function(Boston,index){
    bag.boston <- randomForest(crim~., data = Boston, subset=index,
                               mtry=13,importance=TRUE)
    boston.bag <- predict(bag.boston,newdata=Boston[-index,])
    boston.test <- Boston[-index,"crim"]
    return(mean(boston.bag))
} 

#boost tree function
boost_function <- function(Boston,train){
    boost.boston <-gbm(crim~., data=Boston[train,],distribution = "gaussian",
                       n.trees = 5000,interaction.depth = 4)
    boston.boost <-predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
    boston.test <- Boston[-train,"crim"]
    return(mean(boston.test))  
}

#bumping(bootstrap) method compare result
result_boot  <- boot(data=Boston,statistic = bag_function,R=100)
result_boost <- boot(data=Boston,statistic = boost_function,R=100)
result_boot
result_boost
