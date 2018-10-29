
set.seed(1)       # Set a seed so that results are reproducible
#------------------------data split-----------------------------------------------#
#read dataset and put feature names to it
dataset <- read.csv("data.csv",header=FALSE,sep=";")
names <- read.csv("names.csv",header=FALSE,sep=";")
names(dataset) <- sapply((1:nrow(names)),function(i) toString(names[i,1]))
trainingSize<-3680
# Choose 3680 objects randomly  for training sample data from 4601 objects(vectors):
trainingIndex <- sample(1:4601,trainingSize)
# Divide the data set into training data and test data:
trainingData<-dataset[trainingIndex,]
testingData<-dataset[-trainingIndex,]
# spam_count <- sum(testingData$y == 1)
# ham_count <- sum(testingData$y == 0)


#---------------------------means and standard deviations---------------------------------#
# Create an empty matrix to put feature means/standard deviations in
num_col <- ncol(trainingData)
mean_sd_matrix <- matrix(0, nrow = (num_col - 1), ncol = 4)

# Loop through each column of the dataset and calculate the mean and
# standard deviations of each feature given that it's spam/not spam.
# Place the results in the corresponding columns of 'mean_sd_matrix'.
# Used https://stackoverflow.com/questions/1660124/how-to-sum-a-variable-by-group#1661144
# as a reference.
i <- 1:(num_col-1)
means <- aggregate(trainingData[, i], by=list(trainingData[, num_col]), FUN=mean)#num_col=58
means <- as.matrix(means)
standard_devs <- aggregate(trainingData[, i], by=list(trainingData[, num_col]), FUN=sd)
standard_devs <- as.matrix(standard_devs)

mean_sd_matrix[, 1] <- means[2, (2:num_col)]
mean_sd_matrix[, 2] <- means[1, (2:num_col)]
mean_sd_matrix[, 3] <- standard_devs[2, (2:num_col)]
mean_sd_matrix[, 4] <- standard_devs[1, (2:num_col)]
colnames(mean_sd_matrix) <- c("1","0","1","0")
# So as to not mess up future calculations, if there are any features
# with 0.0 for their means and/or standard deviations, replace those values
# with 0.0001.
# Used https://stackoverflow.com/questions/9439619/replace-all-values-in-a-matrix-0-1-with-0
# as a reference.
mean_sd_matrix[mean_sd_matrix == 0.0] <- 0.0001


#---------------------predictions-------------------------------------------#

# Expects the test set and matrix of feature means and standard deviations.
# Creates a matrix of probabilities where each row represents a data sample.
# Each column represents:
# probability_matrix[i, 1] = probability of that data sample being spam
# probability_matrix[i, 2] = probability of that data sample not being spam


num_samples <- nrow(testingData)
num_cols <- ncol(testingData)

# Stores the results of the probability density function (PDF) P(X|Ci) on each feature for each sample, given that the sample is spam/not spam
conditional_spam <- matrix(0, nrow = num_samples, ncol = (num_cols - 1))#pad with 0
conditional_not_spam <- matrix(0, nrow = num_samples, ncol = (num_cols - 1))
probability_matrix <- matrix(0, nrow = num_samples, ncol = 2)
# Calculate the probability density function (PDF) likehood [Gaussian Function] for each feature of each sample in dataset.
for (i in 1:num_samples)
{
  for (j in 1:(num_cols - 1))
  {
    feature <- testingData[i, j]
    spam_mu <- mean_sd_matrix[j, 1]
    not_spam_mu <- mean_sd_matrix[j, 2]
    spam_sigma <- mean_sd_matrix[j, 3]
    not_spam_sigma <- mean_sd_matrix[j, 4]
    
    conditional_spam[i,j] <- (1/(sqrt(2*pi)*spam_sigma)) * exp(-(((feature - spam_mu)^2)/(2 * spam_sigma^2)))
    conditional_not_spam[i,j] <- (1/(sqrt(2*pi)*not_spam_sigma)) * exp(-(((feature - not_spam_mu)^2)/(2 * not_spam_sigma^2)))
  }
}

# Convert every PDF P(X|Ci) result to its log in order to add the values of each feature rather than multiply.
# This will help avoid buffer overflow.
conditional_spam <- log10(conditional_spam)
conditional_not_spam <- log10(conditional_not_spam)
# Sum the PDF results for each sample and store sum in probability_matrix

colnames(probability_matrix)<- c("conditional_not_spam","conditional_spam")
probability_matrix[, 1] <- rowSums(conditional_not_spam)
probability_matrix[, 2] <- rowSums(conditional_spam)
probability_matrix[probability_matrix == '-Inf'] <- log10(.Machine$double.xmin) # Corrects buffer overflow
# Calculate the spam & non-spam priors
# Used https://www.theanalysisfactor.com/r-tutorial-count/ as a reference
num_spam <- sum(testingData$y == 1)
num_not_spam <- sum(testingData$y == 0)
spam_prior <- num_spam / num_samples
not_spam_prior <- num_not_spam / num_samples
# Take the log of the spam & non-spam priors and add it to their corresponding log sums
#P(X|Ci)*P(Ci)
probability_matrix[, 1] <- probability_matrix[, 1] + log10(spam_prior)
probability_matrix[, 2] <- probability_matrix[, 2] + log10(not_spam_prior)

#---------------------------------result -------------------------------#

# Takes a matrix of spam/non-spam probabilities of each data sample and the test dataset and 
# compares the prediction of each data sample in the probability_matrix with the target 
# in the test data. It then creates a confusion matrix where:
# confusion_matrix[1, 1] = Actual is non-spam and Prediction is non-spam (TN)
# confusion_matrix[1, 2] = Actual is non-spam and Prediction is spam (FP)
# confusion_matrix[2, 1] = Actual is spam and Prediction is non-spam (FN)
# confusion_matrix[2, 2] = Actual is spam and Prediction is spam (TP)
# Returns confusion matrix to the calling routine.

confusion_matrix <- matrix(0, nrow = 2, ncol = 2)
num_samples <- nrow(probability_matrix)
max_col <- ncol(testingData)

for (i in 1:num_samples)
{
  predicted_class <- which.max(probability_matrix[i, ])#the col number of the max(1 or 2)
  actual_class <- testingData[i, max_col] + 1
  confusion_matrix[actual_class, predicted_class] <- confusion_matrix[actual_class, predicted_class] + 1
}
con_matrix_row_names=c("Actual Not Spam","Actual Spam")
con_matrix_col_names=c("Predicted Not Spam","Predicted Spam")
dimnames(confusion_matrix)=list(con_matrix_row_names,con_matrix_col_names)
print("Confusion Matrix")
print(confusion_matrix)
print("***************")

#accuracy
# Expects a confusion matrix as an argument.
# Returns a float representing the accuracy percentage of a confusion matrix: 
# (TP+TN) / (TP+FP+FN+TN) * 100

accuracy <- sum(diag(confusion_matrix))#提取对角线求和
accuracy <- accuracy / sum(confusion_matrix)#精确度=对角线和/矩阵和
accuracy <- accuracy * 100
print("Accuracy:")
print(accuracy)

#Sensitivity: TPR：true positive rate，描述识别出的所有正例占所有正例的比例 
#计算公式为：TPR=TP/ (TP+ FN) 
sensitivity <- confusion_matrix[2,2]/(confusion_matrix[2,2]+confusion_matrix[2,1])
sensitivity <- sensitivity * 100
print("Sensitibity:")
print(sensitivity)
#specificity TNR：true negative rate，描述识别出的负例占所有负例的比例 
#计算公式为：TNR= TN / (FP + TN) 
specifity <- confusion_matrix[1,1]/(confusion_matrix[1,2]+confusion_matrix[1,1])
specifity <- specifity * 100
print("Specifity:")
print(specifity)

#---------------------to be continued------------------------------------------#

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
