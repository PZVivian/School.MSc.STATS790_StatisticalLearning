# ----- PACKAGE & DATA SETUP ----- #
library(tidyverse)
library(GGally)
library(corrplot)
library(e1071)
library(randomForest)
library(kableExtra)

# Load data
marketing_0 <- read_tsv("marketing_campaign.csv")


# ----- DATA TRANSFORMATION ----- #
# Check for missing data
sum(sapply(marketing_0, function(col){ifelse(is.na(col), 1, 0)}))
sum(sapply(marketing_0, function(col){ifelse(sum(col == ""), 1, 0)}))

# Helper table with values to use for imputation
education_vals <- marketing_0 %>%
  group_by(Education) %>% 
  summarize(Imputed_Income = mean(Income, na.rm = TRUE)) %>% 
  arrange(Imputed_Income) %>% 
  mutate(Education_Num = c(1,2,3,4,5))

# Impute missing values in the income column
# Convert education to numerical variable
marketing_1 <- marketing_0 %>% 
  left_join(education_vals, by = join_by(Education == Education)) %>% 
  mutate(Income = ifelse(is.na(Income), Imputed_Income, Income)) %>% 
  select(-c(`ID`, `Education`, `Imputed_Income`)) %>% 
  mutate(Education = `Education_Num`) %>% 
  select(-`Education_Num`)

# Check missing values again
sum(sapply(marketing_1, function(col){ifelse(is.na(col), 1, 0)}))

# Check marital status values
marketing_2 <- marketing_1 %>% 
  group_by(Marital_Status) %>% 
  summarise(count = n())
marketing_2

# Check for constant columns (i.e. with only one value)
which(apply(marketing_1, 2, var)==0)

# Handle non-interpretable values
marketing <- marketing_1 %>%
  mutate(
    Dt_Customer = as.numeric(as.Date(
      paste0(substr(Dt_Customer, 7, 10), 
             substr(Dt_Customer, 3, 6), 
             substr(Dt_Customer, 1, 2))
      )),
    Kidhome = as.integer(Kidhome),
    Teenhome = as.integer(Teenhome),
    Complain = as.integer(Complain),
    AcceptedCmp1 = as.integer(AcceptedCmp1),
    AcceptedCmp2 = as.integer(AcceptedCmp2),
    AcceptedCmp3 = as.integer(AcceptedCmp3),
    AcceptedCmp4 = as.integer(AcceptedCmp4),
    AcceptedCmp5 = as.integer(AcceptedCmp5),
    Complain = as.integer(Complain),
    Response = as.integer(Response),
    Marital_Status = ifelse(Marital_Status == "Alone", 
                            "Single", Marital_Status)) %>% 
  filter(! Marital_Status %in% c("Absurd", "YOLO")) %>% 
  mutate(Single = as.integer(ifelse(Marital_Status == "Single", 1, 0)),
         Married = as.integer(ifelse(Marital_Status == "Married", 1, 0)),
         Together = as.integer(ifelse(Marital_Status == "Together", 1, 0)),
         Divorced = as.integer(ifelse(Marital_Status == "Divorced", 1, 0)),
         Widow = as.integer(ifelse(Marital_Status == "Widow", 1, 0))) %>% 
  select(-c(Marital_Status, Z_CostContact, Z_Revenue))


# Pairs plot of selection of variables
marketingV <- within(marketing, Outcome <- ifelse(Response==1,"Yes", "No"))
ggpairs(data=marketingV[,c(2,3,5,25,31)], aes(colour=Outcome, alpha=0.4))


# Correlation plot
corr_matrix <- cor(marketing)
corrplot(round(corr_matrix,2), method = "number", number.cex=0.75)


# Split data into train and test
set.seed(790)
marketing_3 <- marketing
marketing_3[, -24] <- scale(marketing[, -24])
train.ind <- sample(1:nrow(marketing_3), nrow(marketing_3) / 2)
marketing.train <- marketing_3[train.ind,]
marketing.test <- marketing_3[-train.ind,]
marketing.test.labs <- marketing_3[-train.ind, "Response"]


# Function to harden predictions
harden <- function(probs){
  n <- length(marketing.test$Response)
  pred.labs <- rep(0,n)
  for(i in 1:n){
  	pred.labs[i] <- ifelse(probs[i] < 0.5, 0, 1)
  }
  return(pred.labs)
}


# ----- SVM: LINEAR KERNEL ------ #
# Cross-validation to choose the best cost
set.seed(790)
tune.out <- tune(svm, Response ~ ., data = marketing.train, 
    kernel = "linear",
    ranges = list(
      cost = c(0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.1, 0.5, 1, 5)
    )
  )
  
summary(tune.out)

cost <- c(tune.out$best.parameters)$cost

# Prediction
set.seed(790)
pred <- predict(tune.out$best.model, newdata = marketing.test)

tabSvmLinear <- table(harden(pred), marketing.test.labs$Response)


# ----- SVM: POLYNOMIAL KERNEL ------ #
# Cross-validation to choose the best cost
set.seed(790)
tune.out <- tune(svm, Response ~ ., data = marketing.train, 
    kernel = "polynomial",
    ranges = list(
      cost = c(0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 5)
    )
  )
  
summary(tune.out)

cost <- c(cost, c(tune.out$best.parameters)$cost)

# Prediction
set.seed(790)
pred <- predict(tune.out$best.model, newdata = marketing.test)

tabSvmPoly <- table(harden(pred), marketing.test.labs$Response)


# ----- SVM: RADIAL KERNEL ------ #
# Cross-validation to choose the best gamma and cost
set.seed(790)
tune.out <- tune(svm, Response ~ ., data = marketing.train, 
    kernel = "radial",
    ranges = list(
      cost = c(0.1, 1, 1.5, 2, 2.3, 2.5, 2.8, 3),
      gamma = c(0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 1)
    )
  )
  
summary(tune.out)

cost <- c(cost, c(tune.out$best.parameters)$cost)
gamma <- c("N/A", "N/A", c(tune.out$best.parameters)$gamma)

# Prediction
set.seed(790)
pred <- predict(tune.out$best.model, newdata = marketing.test)

tabSvmRadial <- table(harden(pred), marketing.test.labs$Response)


tuneSummary <- data.frame("Kernel" = c("Linear", "Polynomial", "Radial"), 
                      "Optimal Cost" = cost,
                      "Optimal Gamma" = gamma,
                      check.names = FALSE)
kable(tuneSummary)


# ----- RANDOM FOREST ------ #
# Tune mtry and ntree parameters
set.seed(790)
forest.tune <- tune.randomForest(as.factor(Response) ~ ., 
                                 data = marketing.train, 
                                 mtry = 1:10, 
                                 ntree = 1:80,
                                 tunecontrol = 
                                   tune.control(sampling = "cross",
                                                cross=5),
                                 type = "class")
summary(forest.tune)
plot(forest.tune)

# Apply the random forest based on the optimal mtry and ntree values
set.seed(790)
forest.marketing <- randomForest(as.factor(Response) ~ ., data = marketing.train,
                             mtry=10, ntree=43, importance=TRUE, type="class")
forest.marketing

varImpPlot(forest.marketing, main="")

# Predictions
set.seed(790)
forest.pred <- predict(forest.marketing, marketing.test, type="class")
tabForest <- table(forest.pred, marketing.test.labs$Response)
tabForest


# ----- RESULTS ----- #
method <- c("SVM Linear Kernel", "SVM Polynomial Kernel", 
            "SVM Radial Kernel", "Random Forest")
crand <- c(classAgreement(tabSvmLinear)$crand,
           classAgreement(tabSvmPoly)$crand,
           classAgreement(tabSvmRadial)$crand,
           classAgreement(tabForest)$crand)
misclass <- c(1-classAgreement(tabSvmLinear)$diag, 
              1-classAgreement(tabSvmPoly)$diag, 
              1-classAgreement(tabSvmRadial)$diag,
              1-classAgreement(tabForest)$diag)
summary <- data.frame("Method" = method, 
                      "Adjusted Rand Index" = round(crand,5),
                      "Misclassification Error Rate" = round(misclass,5),
                      check.names = FALSE)
kable(summary)


# VARIABLE DESCRIPTION
category <- c("People", "", "", "", "Products", "Promotion", "", "Place", "")
variables <- c("ID, birth year, education level, marital status, income, number of children",
               "in household, number of teenagers in household, enrolment date with company,",
               "number of days since last purchase, whether the customer complained over the", 
               "last two years",
               "Amount spent on wine, fruits, meat, fish, sweets, gold over the last 2 years",
               "Number of purchases made with discount, whether the customer",
               "accepted the offer for the past 5 campaigns",
               "Number of website visits in the last month, number of purchases",
               "made on the website, catalog, and in-store")
desc <- data.frame("Category" = category, 
                   "Variables" = variables,
                   check.names = FALSE)
kable(desc)

