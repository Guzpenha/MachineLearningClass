library(ggplot2)
require(gridExtra)

df = read.csv("/home/guzpenha/personal/MachineLearningClass/tp1/data/experiments_results_old.csv",sep = ";")
df$numHiddenUnits = as.character(df$numHiddenUnits)

ggplot(df, aes(x=epoch,y=crossEntropy,color=numHiddenUnits)) + 
  geom_line() +
  facet_grid(learningRate~gradientMethod)+ ggtitle("Empirical Error for each epoch")

df2 = read.csv("/home/guzpenha/personal/MachineLearningClass/tp1/data/experiments_results.csv",sep = ";")
df2$numHiddenUnits = as.character(df2$numHiddenUnits)
ggplot(df2, aes(x=updateIteration,y=crossEntropy,color=numHiddenUnits)) + 
  geom_line() +
  facet_grid(learningRate~gradientMethod) + ggtitle("Empirical Error for each weight update")

df2 =df2[df2$learningRate==0.01,]
ggplot(df2, aes(x=updateIteration,y=crossEntropy,color=numHiddenUnits)) + 
  geom_line() +
  facet_grid(learningRate~gradientMethod) + ggtitle("Empirical Error")

df3 = read.csv("/home/guzpenha/personal/MachineLearningClass/tp1/data/experiments_results_batch_size.csv",sep = ";")
df3$batchSize = as.character(df3$batchSize)
ggplot(df3, aes(x=epoch,y=crossEntropy,color=batchSize)) +
 geom_line() + ggtitle("Empirical Error of Mini-Batch GD")

