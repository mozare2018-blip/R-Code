################################################
############  PAPER  ###########################
################################################

Return=read.csv("Return.csv")
attach(Return)
head(Return)
Return=as.data.frame(Return)
######### fit the NN #############
maxs = apply(Return, 2, max) 
mins = apply(Return, 2, min)
scaled = as.data.frame(
  scale(Return,center = mins, scale = maxs - mins))

#index = sample(1:nrow(scaled),round(0.75*nrow(scaled)))
train_ = scaled[1:868,]
test_ = scaled[869:1152,]

library(neuralnet)
n = names(train_)
f = as.formula(
  paste("ret ~", paste(n[!n %in% "ret"],
                      collapse = " + ")))

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

nn = neuralnet(
  f,data=train_,hidden=c(21),act.fct = sigmoid,linear.output=T)

#print(nn)
#plot(nn)
#gwplot(nn)
####Predicting using the neural network

pr.nn = compute(nn,test_[,1:10])

pr.nn_ = pr.nn$net.result*(max(Return$ret)-min(Return$ret))+min(Return$ret)
test.r = (test_$ret)*(max(Return$ret)-min(Return$ret))+min(Return$ret)
MSE.nn = sum((test.r - pr.nn_)^2)/nrow(test_)
print(paste(MSE.nn))
######### fit the AR(1) #############
train1 = Return$ret[1:868]
test1 = Return$ret[869:1152]

ar1_model <- arima(train1, order = c(1, 0, 0))
summary(ar1_model)


library(forecast)
forecast_values <- forecast(ar1_model, h = length(test1))
mse.AR1 <- mean((test1 - forecast_values$mean)^2)
print(paste(mse.AR1))
######### fit the LASSO #############
library(glmnet)   # For Lasso regression
library(caret)    # For data preprocessing (optional)

X_train <- Return[1:868,1:10]
y_train <- Return[1:868,11]
X_test <- Return[869:1152,1:10]
y_test <- Return[869:1152,11]

# Standardize predictors (mean=0, sd=1)
preprocess_params <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preprocess_params, X_train)
X_test_scaled <- predict(preprocess_params, X_test)
# Find optimal lambda via 10-fold CV
cv_lasso <- cv.glmnet(
  x = as.matrix(X_train_scaled),
  y = y_train,
  alpha = 1,          # alpha=1 for Lasso
  nfolds = 10
)

# Best lambda (minimizes MSE)
best_lambda <- cv_lasso$lambda.min

final_lasso <- glmnet(
  x = X_train_scaled,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# View coefficients (some may be zero)
coef(final_lasso)

# Predict on scaled test data
y_pred <- predict(final_lasso, newx =as.matrix(X_test_scaled))

# Calculate MSE
mse.lasso <- mean((y_test - y_pred)^2)
print(paste(mse.lasso))

#plot(y_test, y_pred, main = "Actual vs Predicted")
#abline(0, 1, col = "red")  # Perfect prediction line
######### fit the Random Forest #############
library(randomForest)
X_train <- Return[1:868,1:10]
y_train <- Return[1:868,11]
X_test <- Return[869:1152,1:10]
y_test <- Return[869:1152,11]

# Load data
train_data <- Return[1:868,]
test_data <- Return[869:1152,]

# Split into X and y
X_train <- train_data[, 1:10]
y_train <- train_data$ret
X_test <- test_data[, 1:10]
y_test <- test_data$ret

# Train model
set.seed(123)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  importance = TRUE
)

# Predict and compute MSE
y_pred <- predict(rf_model, X_test)
mse.RF <- mean((y_test - y_pred)^2)
print(paste(mse.RF))

# Feature importance
varImpPlot(rf_model,main = "Random Forest Model")

######### fit the Boosted trees #############
library(xgboost)
train_data <- Return[1:868,]
test_data <- Return[869:1152,]

# Separate predictors (X) and response (y)
X_train2 <- as.matrix(train_data[, 1:10])
y_train2 <- train_data$ret
X_test2 <- as.matrix(test_data[, 1:10])
y_test2 <- test_data$ret


# Convert data to DMatrix (xgboost format)
dtrain <- xgb.DMatrix(data = X_train2, label = y_train2)

# Set parameters
params <- list(
  objective = "reg:squarederror",  # For regression
  eta = 0.1,                      # Learning rate
  max_depth = 6                   # Tree depth
)

# Train model
model2 <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

preds2 <- predict(model2, newdata = X_test2)

# Calculate MSE
mse.BT <- mean((y_test2 - preds2)^2)
print(paste(mse.BT))

######### fit the NNAR #############

library(nnet)
library(forecast)
library(thief)
library(forecastHybrid)

train_data5 <- Return$ret[1:1100]
test_data5 <- Return$ret[1101:1152]



fit1=nnetar(train_data5,p =15, size = 10,act.fac="tanh")
fcast=forecast(fit1,h=length(test_data5))
plot(fcast)
mse.NNAR <- mean((test_data5 - fcast$mean)^2)
print(paste(mse.NNAR))


#######   SVR   ###########################

library(e1071)
library(ggplot2)
library(Metrics)  # for MSE

X_train <- as.data.frame(Return[1:1100,])
X_test <- as.data.frame(Return[1101:1152,])


# Train SVR with RBF kernel
svr_model <- svm(ret ~ ., 
                 data = X_train, 
                 type = "eps-regression", 
                 kernel = "radial", 
                 cost = 100, 
                 epsilon = 0.1, 
                 gamma = 0.1)

# Predict on test data
y_pred <- predict(svr_model, newdata = X_test)

# Calculate MSE
mse_value <- mse(X_test$ret, y_pred)

print(mse_value)






1-(0.0019/0.0029)




























######### Test for nanlinearity ###################
model=lm(ret~.,Return)
library(lmtest)

# Perform RESET test with default settings (powers=2:3)
reset_test <- resettest(model, power = 2:3, type = "fitted")

# View results
print(reset_test)


cor(Return)


dev.new()


# Define model names and R²_oos values
model_names <- c("LASSO", 
                 "Boosted Trees", 
                 "Neural Network", 
                 "NN Autoregressive", 
                 "Boosted Neural Network", 
                 "Graph Neural Network", 
                 "Random Forest")

r2_oos <- c(0.14, 0.29, 0.19, 0.34, 0.10, 0.29, 0.36)

# Create horizontal bar plot
library(ggplot2)

# Create a data frame for plotting
df <- data.frame(
  Model = factor(model_names, levels = rev(model_names)),  # for reverse order
  R2_oos = r2_oos
)

# Plot
ggplot(df, aes(x = R2_oos, y = Model)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = sprintf("%.2f", R2_oos)), 
            hjust = -0.1, size = 4.5) +
  theme_minimal() +
  labs(title = "Out-of-Sample Forecast Performance (R²_oos)",
       x = "R²_oos",
       y = NULL) +
  xlim(0, 0.4) +
  theme(
    text = element_text(family = "Helvetica", size = 12),
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(size = 11)
  )

