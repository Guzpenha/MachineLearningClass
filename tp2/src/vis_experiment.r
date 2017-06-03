require("ggplot2")
require("gridExtra")
df = read.csv("~/personal/MachineLearningClass/tp2/data/senstitivity_n_estimators.csv")
df = df[-1,]

ggplot(df,aes(x=n_estimators,y=AverageAccuracy)) + 
  geom_line() + 
  ylim(0,1)+
  ggtitle("AdaBoostClassifier accuracy by the number of stumps (estimators) on TicTacToe dataset")+
  ylab("Average accuracy of 5 folds cv")

