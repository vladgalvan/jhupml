<style type="text/css">
body, td { font-size: 12px; }
code.r{  font-size: 11px; }
pre {   font-size: 11px }
</style>



## Prediction of Quality of Weight Lifting Exercises with Machine Learning

<p>&nbsp;</p>
#### Executive Summary
The aim of this analysis is to build a model to predict how well people perform weight lifting exercises (WLE) using data from aceloremeters on the belt, forearm, arm, and dumbell from 6 participants and machine learning tools in 'r', mainly the 'caret' package.

[Training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) data were provided. Training data was split in training and testing datasets. Feature selection was made through exploratory analysis and testing, removing near zero covariates, variables with high proportion of missing values, variables considered irrelevant and high correlation variables. Original 160 variables were reduced to 31 predictor and the outcome "Classe". 

Cross validation with 10K fold was used for model selection and optimization, several methods like rf and rpart were tested but gbm (boosting with trees) reported better model accuracy and had a good prediction result against the testing datasets (the result of the split and the 20 test cases in [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv))

The data for this project was provided by
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)


<p>&nbsp;</p>
####Data Split
Original training data was split in training and testing datasets since the original testing data doesn't have the output column "Classe" because it's used for final evaluation of the model.

```r
trainingdata = read.csv("~/r/pml-training.csv"); endtestingdata = read.csv("~/r/pml-testing.csv")
set.seed(1)
intraining<-createDataPartition(trainingdata$classe, p = .75, list = FALSE)
training <- trainingdata[intraining, ]; testing <- trainingdata[-intraining, ]
dim(training); dim(testing)
```

```
## [1] 14718   160
```

```
## [1] 4904  160
```
<p>&nbsp;</p>
####Feature Selection
#####Near Zero Variates
Near zero variables were removed reducing columns from 160 to 108

```r
nzv <- nearZeroVar(training); training <- training[, -nzv]; dim(training)
```

```
## [1] 14718   108
```
#####Irrelevant Variables
The first 6 columns from dataset were removed to be considered irrelevant to explain in general how well an exercise is performed, e.g. user_name who performed the exercise, date/time when the exercise was performed, etc. Columns were reduced from 108 to 102.

```r
summary(trainingdata[1:6])
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp  new_window 
##  28/11/2011 14:14: 1498   no :19216  
##  05/12/2011 11:24: 1497   yes:  406  
##  30/11/2011 17:11: 1440              
##  05/12/2011 11:25: 1425              
##  02/12/2011 14:57: 1380              
##  02/12/2011 13:34: 1375              
##  (Other)         :11007
```

```r
training<-subset(training, select = -c(1,2,3,4,5,6)); dim(training)
```

```
## [1] 14718   102
```
#####Missing Values
Columns with high proportion of missing values were removed, it was considered a better strategy to discard this variables instead of reducing the size of the sample. This was found with the summary function, some examples are shown. Columns were reduced from 102 to 53.

```r
summary(subset(trainingdata,select=c(max_roll_belt,max_picth_belt,min_roll_belt,min_pitch_belt)))
```

```
##  max_roll_belt     max_picth_belt  min_roll_belt     min_pitch_belt 
##  Min.   :-94.300   Min.   : 3.00   Min.   :-180.00   Min.   : 0.00  
##  1st Qu.:-88.000   1st Qu.: 5.00   1st Qu.: -88.40   1st Qu.: 3.00  
##  Median : -5.100   Median :18.00   Median :  -7.85   Median :16.00  
##  Mean   : -6.667   Mean   :12.92   Mean   : -10.44   Mean   :10.76  
##  3rd Qu.: 18.500   3rd Qu.:19.00   3rd Qu.:   9.05   3rd Qu.:17.00  
##  Max.   :180.000   Max.   :30.00   Max.   : 173.00   Max.   :23.00  
##  NA's   :19216     NA's   :19216   NA's   :19216     NA's   :19216
```

```r
training<-subset(training,select = -c(max_roll_belt,max_picth_belt,min_roll_belt,min_pitch_belt,amplitude_roll_belt,amplitude_pitch_belt,var_total_accel_belt,avg_roll_belt,stddev_roll_belt,var_roll_belt,avg_pitch_belt,stddev_pitch_belt,var_pitch_belt,avg_yaw_belt,stddev_yaw_belt,var_yaw_belt,var_accel_arm,max_roll_arm,max_picth_arm,max_yaw_arm,min_roll_arm,min_pitch_arm,min_yaw_arm,amplitude_roll_arm,amplitude_pitch_arm,amplitude_yaw_arm,max_roll_dumbbell,max_picth_dumbbell,min_roll_dumbbell,min_pitch_dumbbell,amplitude_roll_dumbbell,amplitude_pitch_dumbbell,var_accel_dumbbell,avg_roll_dumbbell,stddev_roll_dumbbell,var_roll_dumbbell,avg_pitch_dumbbell,stddev_pitch_dumbbell,var_pitch_dumbbell,avg_yaw_dumbbell,stddev_yaw_dumbbell,var_yaw_dumbbell,max_roll_forearm,max_picth_forearm,min_roll_forearm,min_pitch_forearm,amplitude_roll_forearm,amplitude_pitch_forearm,var_accel_forearm))
dim(training)
```

```
## [1] 14718    53
```
#####Correlated Predictors
Columns were reduced from 53 to 32 (31 predictor and the outcome "Classe") when removing high correlation variables 

```r
set.seed(7); correlationMatrix<-cor(training[,-c(53)]); highlyCorrelated<-findCorrelation(correlationMatrix, cutoff=0.75)
training<-subset(training, select = -c(highlyCorrelated)); dim(training)
```

```
## [1] 14718    32
```
<p>&nbsp;</p>
####Cross Validation
Cross validation with 10K fold was used for model selection and optimization, CV was applied in trControl function in caret.

```r
ctrl <- trainControl(method = "cv", number = 10,allowParallel = TRUE)
```
<p>&nbsp;</p>
####Model Selection
Several methods like rf and rpart were tested but gbm (boosting with trees) reported a better accuracy (0.94578) when fitting the model (in sample accuracy).

```r
modFit1<-train(classe~ .,data=training,method="gbm",verbose=FALSE,trControl= ctrl)
```

```
## Loading required package: plyr
```

```r
print(modFit1)
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    31 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13247, 13247, 13246, 13245, 13246, 13249, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7090672  0.6310721
##   1                  100      0.7752453  0.7154943
##   1                  150      0.8097618  0.7593243
##   2                   50      0.8241647  0.7772893
##   2                  100      0.8807606  0.8491166
##   2                  150      0.9092283  0.8851400
##   3                   50      0.8701623  0.8356118
##   3                  100      0.9242425  0.9041163
##   3                  150      0.9457800  0.9314012
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
<p>&nbsp;</p>
####Prediction Accuracy / Out Of Sample Error
Predictions of the model against the testing datasets were 95% accurate. This is shown in the confusion matrix against the split test dataset. The out of sample error not explained by the model is 5%. The model was also evaluated
against the 20 test cases in [testing](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)) with a similar prediction accuracy.

```r
pred<-predict(modFit1,newdata=testing)
confusionMatrix(pred,testing$classe)$table; confusionMatrix(pred,testing$classe)$overall['Accuracy']
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1360   24    0    3    3
##          B   21  886   41    8   18
##          C    8   37  790   32   11
##          D    5    1   24  755   11
##          E    1    1    0    6  858
```

```
##  Accuracy 
## 0.9480016
```
<p>&nbsp;</p>
####Conclusions
<span style="color:blue">The proposed model predicts with 95% accuracy how well people perform weight lifting exercises (WLE)
</span>. Key steps for the exercise were feature selection which reduced the amount of variables which is very relevant for computer processing when fitting the model and cross validation which is very important to optimize accuracy.
</span>

<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
