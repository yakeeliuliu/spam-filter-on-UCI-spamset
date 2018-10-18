# install.packages('ggplot2')
# install.packages('caret')
# install.packages('e1071', dependencies=TRUE)
#.....................................preprocess dataset.........................#
#read dataset and put feature names to it
dataset <- read.csv("data.csv",header=FALSE,sep=";")
names <- read.csv("names.csv",header=FALSE,sep=";")
names(dataset) <- sapply((1:nrow(names)),function(i) toString(names[i,1]))

#--------------------------------------------k_folds-----------------------------#
k_folds <- 10
folds <- list()

# Split the dataset into a list of data frames where each data frame consists of randomly selected rows from the
# dataset.
# Used https://stats.stackexchange.com/questions/149250/split-data-into-n-equal-groups as a reference
folds <- split(dataset, sample(1:10, nrow(dataset), replace = T))
#result <- matrix(0, nrow = k_folds, ncol = 3) # Stores the accuracy from each fold
result_list <- list()
confusion_matrix_final <- matrix(0, nrow = 0, ncol = 2)
confusion_matrix_list <- list()
#把10份中的一份给testingdata（400多个observations），其余9份给trainingdata（4000多个）
for (q in 1:10)
{
  testingData <- folds[[q]]
  trainingData <- matrix(0, nrow = 0, ncol = 58)
  for (p in 1:10)
  {
    if (p != q)
    {
      trainingData <- rbind(trainingData, folds[[p]])
    }
  }
  
  
  # mean_sd_matrix <- mean_standard_dev(dataset=training_set)
  # probability_matrix <- get_class_predictions(dataset = test_set, mean_sd_matrix = mean_sd_matrix)
  # confusion_matrix <- predict(dataset = test_set, probability_matrix = probability_matrix)
  # Create vector to put classes in:
  
  
  class <- knn(trainingData = trainingData,testingData = testingData, k=3)
  confusion_matrix <- confuma(trainingData = trainingData,testingData = testingData,class = class)
  rownames(confusion_matrix) <- c("actual_non_spam","actual_spam")
  colnames(confusion_matrix) <- c("predicted_non_spam","predicted_spam")
  result_list[[q]] <- get_result(confusion_matrix = confusion_matrix)
  confusion_matrix_final <- rbind(confusion_matrix_final, confusion_matrix)
  print(confusion_matrix)
  #confu_ma <- cbind(confu_ma(,q:q+1), confusion_matrix)
  #accuracy_list[[q]] <- accuracy1
}
# print("confusion matrix of each fold")
colnames(confusion_matrix_final) <- c("predicted_non_span","predicted_spam")
rownames(confusion_matrix_final) <- c(rep(c("actual_non_spam","actual_spam"),k_folds))
print("10 folds of confusion matrix")
print(confusion_matrix_final)

result_matrix <- matrix(unlist(result_list), ncol = 3, byrow = TRUE)
colnames(result_matrix) <- c("accuracy","sensitivity" , "specifity")
result_average <- c(mean(result_matrix[,1]),mean(result_matrix[,2]),mean(result_matrix[,3]))
result_max <- c(max(result_matrix[,1]),max(result_matrix[,2]),max(result_matrix[,3]))
result_min <- c(min(result_matrix[,1]),min(result_matrix[,2]),min(result_matrix[,3]))
result_deviation <- c(sd(result_matrix[,1]),sd(result_matrix[,2]),sd(result_matrix[,3]))
result_matrix <- rbind(result_matrix, result_average,result_max,result_min,result_deviation)
rownames(result_matrix) <- c(1:10,"ave","max","min","dev")
print("result of each fold (%):")
print(result_matrix)



# ------------------Below is the kNN algorithm----------------------------#
knn <- function(trainingData, testingData, k){
# # Choose k:
# k <- 3
class<-rep(0,dim(testingData)[1])#replicate 0 921 times put in class

for(i in 1:dim(testingData)[1])#looping over testing data
{
  # Calculate distances to points in training data:
  #把测试集中的元素按照训练集行数重复,每行57个元素重复3680次，将测试集扩大到训练集行数
  y<-rep(as.matrix(testingData[i,1:57]),dim(trainingData)[1])
  #把y中的元素即测试集和训练集对齐放在ymatrix中
  ymatrix<-matrix(y,nrow=dim(trainingData)[1],ncol=57,byrow=TRUE)
  
  diff<-as.matrix(trainingData[,1:57])-ymatrix
  diff2<-diff^2
  
  dist<-rowsum(t(diff2),rep(dim(diff2)[1],dim(diff2)[2]))#https://stat.ethz.ch/pipermail/r-help/2008-February/153151.html
  
  # Get objects for the observations sorted by the distances:
  distSorted<-sort.int(dist,index.return=TRUE)$ix#范围在1~3680之间，返回记录的位置
  
  # k nearest neighbours vote about class:
  vote<-0
  for(j in 1:k)
  {
    vote<-vote+trainingData[distSorted[j],58]
  }
  
  # Classify points according to votes
  if(vote/k > 0.5) {class[i]<-1}
}
return(class)
}
#---------------------------------confusion_matrix-------------------------------#

# Takes a matrix of spam/non-spam probabilities of each data sample and the test dataset and 
# compares the prediction of each data sample in the probability_matrix with the target 
# in the test data. It then creates a confusion matrix where:
# confusion_matrix[1, 1] = Actual is non-spam and Prediction is non-spam (TN)
# confusion_matrix[1, 2] = Actual is non-spam and Prediction is spam (FP)
# confusion_matrix[2, 1] = Actual is spam and Prediction is non-spam (FN)
# confusion_matrix[2, 2] = Actual is spam and Prediction is spam (TP)
# Returns confusion matrix to the calling routine.
confuma <- function(trainingData,testingData,class){
confusion_matrix <- matrix(0, nrow = 2, ncol = 2)
num_samples <- nrow(testingData)
max_col <- ncol(testingData)

for (i in 1:num_samples)
{
    if(class[i]== 0)#如果是non_spam则落在第一列，否则为spam落在第二列
    {predicted_class <- 1 
    }
    else{
      predicted_class <- 2
    }
  actual_class <- testingData[i, max_col] + 1  #如果是spam则为1+1=2，落在第二行，否则落在第一行
 confusion_matrix[actual_class, predicted_class] <- confusion_matrix[actual_class, predicted_class] + 1
}
return(confusion_matrix)
}

#--------------------------result-------------------------------#

get_result <- function(confusion_matrix){
accuracy <- sum(diag(confusion_matrix))#提取对角线求和
accuracy <- accuracy / sum(confusion_matrix)#精确度=对角线和/矩阵和
accuracy <- accuracy * 100
#Sensitivity: TPR：true positive rate，描述识别出的所有正例占所有正例的比例 
# #计算公式为：TPR=TP/ (TP+ FN) 
sensitivity <- confusion_matrix[2,2]/(confusion_matrix[2,2]+confusion_matrix[2,1])
sensitivity <- sensitivity * 100
# #specificity TNR：true negative rate，描述识别出的负例占所有负例的比例 
# #计算公式为：TNR= TN / (FP + TN) 
specifity <- confusion_matrix[1,1]/(confusion_matrix[1,2]+confusion_matrix[1,1])
specifity <- specifity * 100
three_amigos <- c(accuracy, sensitivity, specifity)
return(three_amigos)
}











#---------------------to be continued------------------------------------------#

k_folds <- 10

set.seed(1)       # Set a seed so that results are reproducible

spambase <- read.csv(file = "spambase.csv", header = FALSE, sep = ",")
spambase <- as.data.frame(spambase)
names(spambase) <- c(1:58)

# Randomize the rows in the dataset.
# Used https://discuss.analyticsvidhya.com/t/how-to-shuffle-rows-in-a-data-frame-in-r/2202 as a reference.
spambase <- spambase[sample(nrow(spambase)), ]

# Perform K-fold cross-validation on dataset
folds <- list()

# Split the dataset into a list of data frames where each data frame consists of randomly selected rows from the
# dataset.
# Used https://stats.stackexchange.com/questions/149250/split-data-into-n-equal-groups as a reference
folds <- split(spambase, sample(1:NUM_FOLDS, nrow(spambase), replace = T))
accuracy_list <- list() # Stores the accuracy from each fold

print("Confusion Matrices for each fold:")
print("TN   |   FP")
print("-----|-----")
print("FN   |   TP")



print("Accuracy (%):")
print(accuracy)
print("Average Accuracy (%):")
print(mean(unlist(accuracy_list)))
print("Max Accuracy (%):")
print(max(unlist(accuracy_list)))
print("Min Accuracy (%):")
print(min(unlist(accuracy_list)))
print("Standard Deviation:")
print(sd(unlist(accuracy_list)))

# trainingDataMatx = as.matrix(trainingData)
# library(e1071)
# nb_model <- naiveBayes(class ~ .,data = trainingDataMatx)
# nb_model
# pred <- predict(nb_model, as.matrix(testingData))

library(ggplot2)
library(caret)
library(e1071)
result1=confusionMatrix(table(class, testingData$y))
result1
