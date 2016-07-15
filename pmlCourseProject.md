# Predicting the Correctness of an Exercise
Amiel Sy  
July 16, 2016  



##Background

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. There are 5 classifications of the exercise:  
-doing the exercise correctly (class A)  
-throwing the elbows to the front (Class B)  
-lifting the dumbbell only halfway (Class C)  
-lowering the dumbbell only halfway (Class D)  
-throwing the hips to the front (Class E)  

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The dataset used in this report is from this publication:   
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 

##Dataset

Please download the datasets and paste them in you working directory before running the code in this report for it to work.

Datasets can be found here: ["http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"]
["http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"]  
Alternatively, datasets can be found from my Github: 

All the variables of training and testing are the same except that testing does not have the classe variable and training does not have the problem_id variable. Our goal for this project is to make a predict class in the training file.

##Data Processing

First, csv files are read.


```r
data = read.csv("pml-training.csv")
submission = read.csv("pml-testing.csv")
```

Libraries are loaded.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```r
library(plyr)
```
Here, columns full of NAs from submission are removed. The first 7 columns are also removed since these variables like names, index, and dates has nothing to do with classe.


```r
variables <- names(submission[,colSums(is.na(submission)) == 0])[8:59]
data<-data[,c(variables, "classe")]
submission<-submission[,c(variables, "problem_id")]
```

Next, the training and testing datasets are created from the data.

```r
set.seed(100)
inTrain = createDataPartition(data$classe, p = 0.8, list = FALSE)
training = data[inTrain,]
testing = data[-inTrain,]
```

##Machine Learning
Here, the random forest algorithm is used. Preprocessing such as center, scaling, and PCA are not used since it does not appear to help in the prediction.

Using the train method, we cross-validate using 4 folds. 

```r
modelrf<-train(classe~., training, trControl=trainControl(method = "cv", number = 4), 
               method="rf")
modelrf
```

```
## Random Forest 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 11776, 11776, 11773, 11772 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9919101  0.9897654
##   27    0.9912734  0.9889604
##   52    0.9865582  0.9829931
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
The cross-validation accuracy for the random forest algorithm is 99.19%.

Next, we use the gbm(boosting) algorithm.


```r
modelgbm<-train(classe~., training, trControl=trainControl(method = "cv", number = 4), 
               method="gbm", verbose=FALSE)
modelgbm
```

```
## Stochastic Gradient Boosting 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 11774, 11774, 11775, 11774 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7461619  0.6778997
##   1                  100      0.8185867  0.7704241
##   1                  150      0.8533659  0.8144395
##   2                   50      0.8552134  0.8165464
##   2                  100      0.9071271  0.8824750
##   2                  150      0.9326702  0.9147993
##   3                   50      0.8970631  0.8696696
##   3                  100      0.9401871  0.9243089
##   3                  150      0.9612073  0.9509183
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
The cross-validation accuracy for the gbm algorithm is 96.12%

Here, the best model (random forest) is used to predict the test set.


```r
predrf<- predict(modelrf, testing)
confusionMatrix(predrf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    2    0    0    0
##          B    0  755    3    0    0
##          C    0    2  679   14    0
##          D    0    0    2  629    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9941          
##                  95% CI : (0.9912, 0.9963)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9926          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9927   0.9782   1.0000
## Specificity            0.9993   0.9991   0.9951   0.9994   1.0000
## Pos Pred Value         0.9982   0.9960   0.9770   0.9968   1.0000
## Neg Pred Value         1.0000   0.9987   0.9985   0.9957   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1925   0.1731   0.1603   0.1838
## Detection Prevalence   0.2850   0.1932   0.1772   0.1608   0.1838
## Balanced Accuracy      0.9996   0.9969   0.9939   0.9888   1.0000
```
The out of sample error is just 1-accuracy=0.59%, which is very good.

##Results

Now, we apply our final model to the submission dataframe to get the final result.

```r
predict(modelrf, submission)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
