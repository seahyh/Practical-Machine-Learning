# Getting and Cleaning Data
### Loading libraries
```{r, message=FALSE, warning=FALSE}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
```
### Obtaining Data
```{r echo=TRUE}
training_data <- read.csv("pml-training.csv")
testing_data <- read.csv("pml-testing.csv")
inTrain <- createDataPartition(training_data$classe, p=0.6, list=FALSE)
myTraining <- training_data[inTrain, ]
myTesting <- training_data[-inTrain, ]
```
### Cleaning Data
```{r echo=TRUE}
# remove near zero variance variables
nzv <- nearZeroVar(myTraining)
myTraining <- myTraining[, -nzv]
myTesting <- myTesting[, -nzv]
# remove mostly NA variables
mostlyNA <- sapply(myTraining, function(x) mean(is.na(x))) > 0.95
myTrainig <- myTraining[, mostlyNA==F]
myTesting <- myTesting[, mostlyNA==F]
# remove identification variables (columns 1 to 5)
myTraining <- myTrainig[, -(1:5)]
myTesting  <- myTesting[, -(1:5)]
```
# Predict Data by various models
### 1. Random forest
```{r echo=TRUE}
modFit <- randomForest(classe ~ ., data=myTraining)
modFit
# Prediction using Random forest
predict <- predict(modFit, myTesting, type="class")
confusionMatrix(myTesting$classe, predict)
```
### 2. Decision tree
```{r echo=TRUE}
modFit_T <- rpart(classe~., myTraining)
# Prediction using Decision tree
predict_T <- predict(modFit_T, myTesting, type="class")
confusionMatrix(myTesting$classe, predict_T)
```
### 3. Generalized Boosted Model (GBM)
```{r, message=FALSE, warning=FALSE}
control_GBM <- trainControl(method = "repeatedcv", number=5, repeats=1)
modFit_GBM <- train(classe~., myTraining, method="gbm", trControl=control_GBM, verbose=FALSE)
```
```{r echo=TRUE}
# Prediction using GBM
predict_GBM <- predict(modFit_GBM, myTesting)
confusionMatrix(predict_GBM, myTesting$classe)
```

# Error and Cross validation
#### Random forest, Dicision tree, and GBM models give us 99.6 %, 75.4 %, and 98.8 % as accuracy, respectively.
#### The expected sample errors for Random forest, Dicision tree, and GBM are 0.4 %, 24.6 %, and 1.2 %, respectively.

# Final test
#### Run the algorithm for the 20 test cases in the test data using the Random forest model, the most accurate model.
```{r echo=TRUE}
predict_test <- predict(modFit, testing_data, type = "class")
predict_test
```
