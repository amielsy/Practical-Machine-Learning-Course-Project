library(caret)
library(randomForest)
library(gbm)

data = read.csv("pml-training.csv")
submission = read.csv("pml-testing.csv")

variables <- names(submission[,colSums(is.na(submission)) == 0])[8:59]
data<-data[,c(variables, "classe")]
submission<-submission[,c(variables, "problem_id")]

set.seed(100)
inTrain = createDataPartition(data$classe, p = 0.1, list = FALSE)
training = data[inTrain,]
testing = data[-inTrain,]

modelrf<-randomForest(training[,1:52],training[,53], ntree=500)
modelrf<-train(classe~., training, trControl=trainControl(method = "cv", number = 4))
predrf<- predict(modelrf, testing)
confusionMatrix(predrf, testing$classe)

predict(modelrf,submission)
