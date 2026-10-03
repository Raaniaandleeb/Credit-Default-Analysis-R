# Credit Default Analysis and Prediction in R

## Project Overview

This project analyzes customer credit card data and builds a logistic regression model to predict whether a customer is likely to default.

The analysis focuses on understanding the relationship between credit card balance, income, student status, and default status.

## Research Question

Can we predict whether a customer will default based on their financial characteristics?

## Dataset

The dataset used is the `Default` dataset from the `ISLR2` R package.

It contains 10,000 customers and the following variables:

- `default` - Whether the customer defaulted (Yes/No)
- `student` - Whether the customer is a student (Yes/No)
- `balance` - Average credit card balance
- `income` - Customer income

## Tools and Packages

- R
- RStudio
- Git
- GitHub
- `ISLR2`
- `ggplot2`
- `pROC`

## Exploratory Data Analysis

The dataset is highly imbalanced:

- 96.67% of customers did not default
- 3.33% of customers defaulted

The analysis showed that balance has a much clearer relationship with default than income.

Customers with higher balances had substantially higher observed default rates. For example, the default rate for customers with balances above 2,000 was about 76.4%.

## Logistic Regression

Three logistic regression models were developed:

1. Balance only
2. Balance + Income
3. Balance + Income + Student Status

Adding income significantly improved the model compared with balance alone.

Adding student status also significantly improved the model.

The final model included:

```text
default ~ balance + income + student