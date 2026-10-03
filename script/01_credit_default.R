# ==================================================
# Credit Default Analysis and Prediction
# ==================================================
# Dataset: ISLR2::Default
# Goal: Explore customer default patterns and build
#       a logistic regression model for prediction.
# ==================================================

# Load required package
install.packages("ISLR2")
library(ISLR2)

# Load the dataset
data <- Default

# Basic inspection
dim(data)
head(data)
summary(data)
str(data)
names(data)
# Q:Can we predict whether a customer will default based on their financial characteristics?

# Check the target variable
table(data$default)
prop.table(table(data$default))

# ---------------------Interpretation----------------------------------------
# The dataset contains 10,000 customers. Most customers did not default
# (96.67%), while only 3.33% defaulted. This shows that the target variable
# is highly imbalanced, which is important to consider when evaluating
# classification models.

# Exploratory Data Analysis
summary(data[, c("balance", "income")])

# ------------------------Interpretation-------------------------------------
# Balance ranges from 0 to about 2,654, with a median of about 824.
# Income ranges from about 772 to 73,554, with a median of about 34,553.
# These summaries describe the overall distributions, but they do not
# show whether balance or income differs between default groups.

# Balance by Default Status
# --------------------------------------------------

library(ggplot2)
ggplot(data, aes(x = default, y = balance)) +geom_boxplot() +labs(title = "Credit Card Balance by Default Status",x = "Default",y = "Balance" )

# --------------------------Interpretation----------------------------
# From the boxplot, customers who defaulted generally have higher
# credit card balances than those who did not default. The median
# balance for the default group is around 1,800, while it is around
# 800 for the non-default group. There is not much overlap between
# the two groups, so balance looks like an important variable for
# predicting default.

# Income by Default Status
ggplot(data, aes(x = default, y = income)) +geom_boxplot() +labs(title = "Income by Default Status",x = "Default",y = "Income")

#---------------------------- Interpretation---------------------------------
# The income distributions for customers who defaulted and those who did
# not default are quite similar. The median income is around 35,000 for
# the non-default group and around 32,000 for the default group. There
# is a lot of overlap between the two groups, so income does not show
# as clear a difference as balance does.

# Default Status by Student Status
ggplot(data, aes(x = student, fill = default)) +geom_bar(position = "fill") +labs(title = "Default Rate by Student Status",x = "Student",y = "Proportion",fill = "Default")

# ---------------------Interpretation----------------------------------
# The default rates for students and non-students look quite similar.
# Students appear to have a slightly higher proportion of defaults,
# but the difference is small. So, student status does not seem to
# separate the two default groups as clearly as balance does.

# Balance vs Income by Default Status
ggplot(data, aes(x = income, y = balance, color = default)) +geom_point(alpha = 0.5) +labs(title = "Balance vs Income by Default Status",x = "Income",y = "Balance",color = "Default")

#----------------------- Interpretation---------------------------------
# Most customers have balances below about 1,500, with income spread
# across a wide range. The default cases are much fewer than the
# non-default cases, but they appear more often at higher balance
# values. There is no clear pattern between income and default status.
# Overall, balance seems to show a stronger relationship with default
# than income.

# Default Rate by Balance
data$balance_group <- cut(data$balance,breaks = c(0, 500, 1000, 1500, 2000, Inf),labels = c("0-500", "500-1000", "1000-1500", "1500-2000", "2000+"))
default_rate <- aggregate(as.numeric(data$default == "Yes"),by = list(Balance = data$balance_group),FUN = mean)
names(default_rate)[2] <- "Default_Rate"
default_rate

# --------------------Interpretation------------------------------------
# The default rate increases strongly as the balance increases.
# Customers with balances below 1,000 have very low default rates,
# while the rate becomes much higher for balances above 1,500.
# The 2000+ group has a default rate of about 76.4%, which is much
# higher than the other groups. This shows a strong relationship
# between balance and default status.

ggplot(default_rate, aes(x = Balance, y = Default_Rate)) +geom_col() +labs(title = "Default Rate by Balance Group",x = "Balance",y = "Default Rate")

# --------------------------Interpretation-----------------------------
# The default rate increases strongly as the balance increases.
# Customers with balances below 1,000 have very low default rates,
# while the rate becomes much higher for balances above 1,500.
# The 2000+ group has a default rate of about 76.4%, which is much
# higher than the other groups. This shows a strong relationship
# between balance and default status.

# Logistic Regression
# --------------------------------------------------

# We use logistic regression because the target variable
# (default) has two possible outcomes: Yes and No.

logistic_model <- glm(default ~ balance,data = data,family = binomial)
summary(logistic_model)

#-------------------- Interpretation-------------------------------------
# The logistic regression shows a significant positive relationship
# between balance and default status (p < 0.001). This means that
# customers with higher credit card balances are more likely to default.
# The result agrees with what we observed in the earlier boxplot and
# default-rate analysis, where higher balances were linked with higher
# default rates.

# why: Odds ratios make logistic regression coefficients easier to interpret.
# Odds Ratio
exp(coef(logistic_model))

# -------------------Interpretation------------------------------
# The odds ratio for balance is 1.0055, meaning that for every
# 1-unit increase in balance, the odds of default increase by
# about 0.55%.

# Odds Ratio for a 100-Unit Increase in Balance
# Why: A 100-unit increase gives a more meaningful interpretation of balance.
exp(100 * coef(logistic_model)["balance"])

#--------------------------- Interpretation--------------------------------
# For every 100-unit increase in balance, the odds of default increase
# by about 73.3%. This shows that higher balances are strongly associated
# with a higher chance of default.

# Logistic Regression with Balance and Income


# Why: We add income to see whether it provides additional information
# about default after accounting for balance.
logistic_model2 <- glm(default ~ balance + income,data = data,family = binomial)
summary(logistic_model2)

# ----------------------Interpretation---------------------------------------
# Both balance and income are significantly related to default status
# (p < 0.001). Balance has a positive coefficient, meaning higher
# balances are associated with higher odds of default. Income also has
# a positive coefficient, but its effect is much smaller than balance.
# Adding income reduces the residual deviance, showing that it provides
# some additional information about default.

# Model Comparison
# Why: We compare the models to check whether adding income
# significantly improves the model beyond balance alone.
anova(logistic_model, logistic_model2, test = "Chisq")

#----------------------- Interpretation----------------------------------
# Adding income significantly improves the logistic regression model
# compared with using balance alone (p < 0.001). Therefore, income
# provides additional information for predicting default.

# Logistic Regression with Balance, Income and Student Status
# Why: We add student status to check whether it provides additional
# information about default after accounting for balance and income.
logistic_model3 <- glm(default ~ balance + income + student,data = data,family = binomial)
summary(logistic_model3)

# ------------------------------Interpretation----------------------------
# Balance is still strongly related to default status (p < 0.001).
# Income is no longer significant after accounting for balance and
# student status (p = 0.712). Student status is significant (p = 0.006),
# while the residual deviance decreases slightly after adding student.

# Model Comparison
# Why: We compare the models to check whether adding student status
# significantly improves the model.
anova(logistic_model2, logistic_model3, test = "Chisq")

#--------------------------- Interpretation----------------------------------
# Adding student status significantly improves the logistic regression
# model (p = 0.006). Therefore, student status provides additional
# information for predicting default after accounting for balance and income.

# Odds Ratios for Final Model
# --------------------------------------------------

# Why: Odds ratios make the logistic regression coefficients
# easier to interpret.
exp(coef(logistic_model3))

#------------------- Interpretation---------------------------------------
# Balance has an odds ratio slightly above 1, showing that higher
# balances are associated with higher odds of default. The odds ratio
# for student status is 0.524, meaning students have about 47.6% lower
# estimated odds of default than non-students after accounting for
# balance and income. Income has an odds ratio very close to 1 and was
# not statistically significant in the final model.

# Model Evaluation: Confusion Matrix
# Why: The confusion matrix shows how well the model classifies
# default and non-default customers.
predicted_prob <- predict(logistic_model3, type = "response")
predicted_class <- ifelse(predicted_prob > 0.5, "Yes", "No")
predicted_class <- factor(predicted_class,levels = levels(data$default))
confusion_matrix <- table(Predicted = predicted_class,Actual = data$default)
confusion_matrix

# Classification Metrics
# Why: These metrics give a more complete view of model performance
# than accuracy alone.
TN <- confusion_matrix["No", "No"]
FP <- confusion_matrix["Yes", "No"]
FN <- confusion_matrix["No", "Yes"]
TP <- confusion_matrix["Yes", "Yes"]
accuracy <- (TP + TN) / sum(confusion_matrix)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
accuracy
sensitivity
specificity
precision

#----------------------- Interpretation-----------------------------------
# The model has an accuracy of 97.32%, meaning it correctly classifies
# most customers overall. Its specificity is very high at 99.59%, so it
# is very good at identifying customers who do not default. However,
# sensitivity is only 31.53%, meaning the model misses many customers
# who actually default. Precision is 72.41%, meaning that about 72% of
# customers predicted to default actually defaulted. Because the dataset
# is highly imbalanced, accuracy alone does not give a complete picture
# of model performance.

# ROC Curve and AUC
# Why: ROC and AUC evaluate how well the model separates default
# and non-default customers across different classification thresholds.
install.packages("pROC")
library(pROC)
roc_curve <- roc(data$default, predicted_prob)
auc_value <- auc(roc_curve)
auc_value
plot(roc_curve,main = "ROC Curve for Credit Default Model")

# --------------------------Interpretation----------------------------------
# The ROC curve shows that the model separates default and non-default
# customers quite well. The AUC is 0.9496, which indicates strong
# discrimination. However, the earlier sensitivity result shows that
# the 0.5 classification threshold still misses many actual defaults.

# Model Evaluation at a Lower Threshold
# Why: A lower threshold can help detect more customers who are likely
# to default, which may improve sensitivity.
predicted_class_20 <- ifelse(predicted_prob > 0.20,"Yes","No")
predicted_class_20 <- factor(predicted_class_20,levels = levels(data$default))
confusion_matrix_20 <- table(Predicted = predicted_class_20,Actual = data$default)
confusion_matrix_20

# Classification Metrics at 0.20 Threshold

# Why: These metrics show how the lower threshold changes the model's performance.

TN_20 <- confusion_matrix_20["No", "No"]
FP_20 <- confusion_matrix_20["Yes", "No"]
FN_20 <- confusion_matrix_20["No", "Yes"]
TP_20 <- confusion_matrix_20["Yes", "Yes"]
accuracy_20 <- (TP_20 + TN_20) / sum(confusion_matrix_20)
sensitivity_20 <- TP_20 / (TP_20 + FN_20)
specificity_20 <- TN_20 / (TN_20 + FP_20)
precision_20 <- TP_20 / (TP_20 + FP_20)
accuracy_20
sensitivity_20
specificity_20
precision_20

#-------------------------- Interpretation--------------------------------
# Lowering the threshold from 0.50 to 0.20 increases sensitivity
# from 31.53% to 60.96%, meaning the model detects more actual defaults.
# However, specificity decreases from 99.59% to 97.13%, and precision
# decreases from 72.41% to 42.29%. This shows the trade-off between
# detecting more defaults and producing more false positives.

# Predicting Default for New Customers
# Why: Predictions show how the trained model can be used on new customer data.
new_customers <- data.frame(balance = c(500, 1200, 1800, 2200),income = c(30000, 35000, 40000, 45000),student = c("No", "No", "Yes", "No"))
new_customers$default_probability <- predict(logistic_model3,newdata = new_customers,type = "response")
new_customers

#---------------------- Interpretation-------------------------------------
# The model gives very low default probabilities to the first two
# customers, while the third customer has a probability of about 25.6%.
# The fourth customer has a much higher probability of about 86.9%.
# Overall, the predictions show that customers with higher balances
# tend to receive higher estimated probabilities of default.

# Final Default Predictions
# Why: Converting probabilities into Yes/No classes makes the predictions
# easier to interpret for a classification problem.
new_customers$predicted_default <- ifelse(new_customers$default_probability > 0.20,"Yes","No")
new_customers

# ---------------------Interpretation-------------------------------------
# Using the 0.20 threshold, the first two customers are predicted
# not to default, while the third and fourth customers are predicted
# to default. The fourth customer has the highest estimated default
# probability at about 86.9%.

# Final Visualization: Default Probability vs Balance
# Why: This plot visually shows how the model's predicted probability
# of default changes as customer balance increases.
prediction_data <- data.frame(balance = seq(min(data$balance),max(data$balance),length.out = 200),income = median(data$income),student = "No")
prediction_data$default_probability <- predict(logistic_model3,newdata = prediction_data,type = "response")
ggplot(prediction_data, aes(x = balance, y = default_probability)) +geom_line() +labs(title = "Predicted Probability of Default",x = "Balance",y = "Predicted Probability of Default")

#------------------------ Interpretation-----------------------------------
# The predicted probability of default stays very low at lower balance
# levels. It starts increasing more quickly around balances of 1,500 to
# 2,000 and becomes very high at balances above 2,000. This supports
# the earlier finding that balance is an important predictor of default.
