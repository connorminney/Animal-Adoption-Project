# Animal Shelter Outcommes
# Developed by Connor Minney & Kaleb Tucker

# Load Packages
pacman::p_load(dplyr, stringr, rpart, rpart.plot, mosaic, FNN,fastDummies, e1071, nnet, corrplot, caret, arm, brnn, LiblineaR, sampling, stringr, varhandle)
options(scipen = 10)

set.seed(42)

# Import Data
asdf <- read.csv("C:\\Users\\conno\\OneDrive\\Desktop\\Machine Learning Project\\aac_intakes_outcomes.csv")

# Subset to relevant fields
asdf = asdf[,c(5:6,8,22:24,26:27,31)]

# Split sex and neutered status into separate columns
split_df = data.frame(str_split_fixed(asdf$sex_upon_outcome, " ", 2))
asdf = merge(asdf, split_df, by.x = 0, by.y = 0)
asdf$neutered = asdf$X1
asdf$male = asdf$X2

# Convert the neutered status and sex to binary values
asdf$neutered = gsub('Neutered', 1, asdf$neutered)
asdf$neutered = gsub('Spayed', 1, asdf$neutered)
asdf$neutered = gsub('Intact', 0, asdf$neutered)
asdf$male = gsub('Male', 1, asdf$male)
asdf$male = gsub('Female', 0, asdf$male)

# Subset to just dogs
asdf = subset(asdf, asdf$animal_type == 'Dog')

# Subset to just adoptions and euthanizations
asdf = subset(asdf, asdf$outcome_type %in% c("Adoption", "Euthanasia"))

# Remove irrelevant columns
asdf = asdf[, c(2, 4, 6:10, 13:14)]

# Convert dependent to binary
asdf$outcome_type = gsub('Adoption', 1, asdf$outcome_type)
asdf$outcome_type = gsub('Euthanasia', 0, asdf$outcome_type)

# Create a flag to determine if the animal is a "mix" or not (i.e., Rottweiler vs. Rottweiler Mix)  
asdf <- dplyr::mutate(asdf, Mix = ifelse(stringr::str_detect(breed, "Mix") == TRUE, 1,0))

# Remove Mix and Second Breed if indicated by "/" (i.e., Labrador Retriever / Pit Bull -> Labrador Retriever w/ Mix Flag)
asdf$breed <- asdf$breed %>% str_replace("Mix", "")
asdf$breed <- asdf$breed %>% str_replace("/.+", "")


# Convert the categorical variables to dummies
breed = data.frame(to.dummy(asdf$breed, 'Breed'))
asdf = merge(asdf, breed, by.x = 0, by.y = 0)

condition = data.frame(to.dummy(asdf$intake_condition, 'condition'))
asdf = merge(asdf, condition, by.x = 0, by.y = 0)

type = data.frame(to.dummy(asdf$intake_type, 'type'))
asdf = merge(asdf, type, by.x = 0, by.y = 0)

# Drop categorical columns after dummy conversion
asdf = asdf[ , -which(names(asdf) %in% c("breed","color", "intake_condition", "intake_type", "Row.names", "Row.names.1"))]

# Drop duplicated columns
asdf <- asdf[, !duplicated(colnames(asdf))]

# Convert everything to numeric
asdf<- data.frame(lapply(asdf,as.numeric))

# Create correlation matrix
correlations <- data.frame(cor(asdf))

# Drop variables with correlation < +/-.05
asdf = asdf[ , which(names(asdf) %in% c("neutered","male","outcome_type", "condition.Normal", "type.Stray", "type.Euthanasia_Request", "age_upon_intake_.years.", "condition.Injured", "condition.Sick", "condition.Aged", "type.Public_Assist", "Mix"))]

# Run stepwise
nullmodel <- glm(outcome_type ~ ., data = asdf, family = binomial )
stepmodel <- step(nullmodel)
summary(stepmodel)

# Drop variables that were not identified as relevant by the stepwise
asdf = asdf[ , which(names(asdf) %in% c("neutered","male","outcome_type", "type.Stray", "type.Euthanasia_Request", "age_upon_intake_.years.", "condition.Injured", "condition.Sick", "type.Public_Assist", "Mix"))]


# Remove any record where fields are incomplete
asdf <- na.omit(asdf)

# Resampling

# Segregate Adoptions & Euthanasia into separate dataframes
adoptions <- asdf[asdf$outcome_type == 1, ]
euthanasia <- asdf[asdf$outcome_type == 0, ]

# Create Random Sample of Rows
adoptionrows <- sample(nrow(adoptions))
euthanasiarows <- sample(nrow(euthanasia))  

# Reduce Adoptions to match euthanizations
adoptions <- adoptions[adoptionrows, ]
adoptions <- adoptions[0:nrow(euthanasia), ] 

asdf1 <- rbind(adoptions, euthanasia)
remove(adoptions)
remove(euthanasia)

asdf1$outcome_type <- as.factor(asdf1$outcome_type)

# Data Pre-Processing

# Splitting Data 
# Split the Data into Training, Validation, and Testing Sets for TVT Validations
tvt <- sample(c('train','valid','test'),size = nrow(asdf1),replace = TRUE, prob = c(0.6,0.3,0.1))
asdf1 <- mutate(asdf1,tvt)

asdf1_train <- filter(asdf1, tvt == 'train') %>% dplyr::select(-tvt)
asdf1_valid <- filter(asdf1, tvt == 'valid')  %>% dplyr::select(-tvt)
asdf1_test <- filter(asdf1, tvt == 'test')  %>% dplyr::select(-tvt)

# Split the Data into Training and Testing for KFold Validations
kfold <- sample(c('train', 'test'),size = nrow(asdf1),replace = TRUE, prob = c(0.8,0.2))
asdf1 <- mutate(asdf1,kfold)

asdf1_kfold_train <- filter(asdf1, kfold == 'train') %>% dplyr::select(-kfold, -tvt)
asdf1_kfold_test <- filter(asdf1, kfold == 'test')  %>% dplyr::select(-kfold, -tvt)


# Normalization for KNN & SVM
asdf1_data_norm <- mutate(asdf1, 
                          age_upon_intake_.years. = (age_upon_intake_.years.-min(age_upon_intake_.years.))/(max(age_upon_intake_.years.)-min(age_upon_intake_.years.)))

asdf1_norm_train <- filter(asdf1_data_norm, tvt == 'train') %>% dplyr::select(-tvt, -kfold)
asdf1_norm_valid <- filter(asdf1_data_norm, tvt == 'valid')  %>% dplyr::select(-tvt, -kfold)
asdf1_norm_test <- filter(asdf1_data_norm, tvt == 'test')  %>% dplyr::select(-tvt, -kfold)

asdf1_norm_kfold_train <- filter(asdf1_data_norm, kfold == 'train') %>% dplyr::select(-kfold, -tvt)
asdf1_norm_kfold_test <- filter(asdf1_data_norm, kfold == 'test')  %>% dplyr::select(-kfold, -tvt)


################################################################################

# End of Connor's Updates

################################################################################    
# Build the Models

# Logistic Regression - TVT
# Notes - w/ Logistic Regressions, we were unable to apply the model to the Validation or Test Data as there were new factors in 
# validation & test which threw the regression. KFold more appropriate at this time. 
logistic_model <- glm(outcome_type ~ ., data = asdf1_train, family = binomial)
summary(logistic_model)

# Validation Predictions
logistic_predictions <- mutate(asdf1_valid, validation_predictions = predict(logistic_model, asdf1_valid, type="response") %>% round())
logistic_predictions$validation_predictions <- as.integer(logistic_predictions$validation_predictions)

# Validation Accuracy
mean(~(outcome_type == validation_predictions), data = logistic_predictions)
# Validation Accuracy - 83.54978%

# Test Predictions
logistic_predictions <- mutate(asdf1_test, test_predictions = predict(logistic_model, asdf1_test, type="response") %>% round())
logistic_predictions$test_predictions <- as.integer(logistic_predictions$test_predictions)

# Test Accuracy
mean(~(outcome_type == test_predictions), data = logistic_predictions)
# Test Accuracy - 78.01%

# Logistic Regression - KFold
# Model - 10 Folds 
logistic_kfold <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = 'bayesglm', trControl = trainControl("cv", number = 10 ))     

# Model Summary
logistic_kfold

# Model Predictions & Accuracy 
asdf1_logistic_kfold_predictions <- mutate(asdf1_kfold_test, model_predictions = predict(logistic_kfold, newdata = asdf1_kfold_test))
mean(~(outcome_type == model_predictions), data = asdf1_logistic_kfold_predictions )
# Model Accuracy 81.3%



#--------------------------------------------------------------------------#

# Stepwise Regression - TVT 
nullmodel <- glm(outcome_type ~ 1, data = asdf1_train, family = binomial)
stepmodel <- step(nullmodel, scope=formula(logistic_model))
summary(stepmodel)

# Validation Predictions & Accuracy
stepwise_validation  <- mutate(asdf1_valid, validation_predictions = round(predict(stepmodel, newdata = asdf1_valid, type = "response"),0))
mean(~(outcome_type == validation_predictions), data = stepwise_validation)
# Validation Data Accuracy - X%

# Test Predictions
stepwise_predictions <- mutate(asdf1_test, test_predictions = predict(stepmodel, asdf1_test, type="response") %>% round())
stepwise_predictions$test_predictions <- as.integer(stepwise_predictions$test_predictions)

# Test Accuracy
mean(~(outcome_type == test_predictions), data = stepwise_predictions)
# Test Accuracy - 70.92%




# Stepwise Regression - KFold

# Model - 10 Folds 
stepwise_kfold <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = 'glmStepAIC', trControl = trainControl("cv", number = 10 ))     

# Model Summary
stepwise_kfold

# Model Predictions & Accuracy 
asdf1_stepwise_kfold_predictions <- mutate(asdf1_kfold_test, model_predictions = predict(stepwise_kfold, newdata = asdf1_kfold_test))
mean(~(outcome_type == model_predictions), data = asdf1_logistic_kfold_predictions )
# Model Accuracy X%

#--------------------------------------------------------------------------#

# Decision Tree - TVT    

# Model
tree_model <- rpart(outcome_type ~ ., data = asdf1_train, method = "class")

# CP - .01
tree_model_prun_01 <- prune(tree_model, cp = 0.01)
rpart.plot(tree_model_prun_01,roundint=FALSE,nn=TRUE,extra=4)

# Validation Predictions & Accuracy
tree_prun_01_validate  <- mutate(asdf1_valid, validation_predictions = predict(tree_model_prun_01, newdata = asdf1_valid, type = "class"))
mean(~(outcome_type == validation_predictions), data = tree_prun_01_validate)
# Validation Accuracy 83.8%

# Test Predictions & Accuracy
tree_prun_01_test  <- mutate(asdf1_test, test_predictions = predict(tree_model_prun_01, newdata = asdf1_test, type = "class"))
mean(~(outcome_type == test_predictions), data = tree_prun_01_test)
# Validation Accuracy 84.21%


# CP - .03
tree_model_prun_03 <- prune(tree_model, cp = 0.03)
rpart.plot(tree_model_prun_03,roundint=FALSE,nn=TRUE,extra=4)

# Validation Predictions & Accuracy 
tree_prun_03_validate  <- mutate(asdf1_valid, validation_predictions = predict(tree_model_prun_03, newdata = asdf1_valid, type = "class"))
mean(~(outcome_type == validation_predictions), data = tree_prun_03_validate)
# Validation Accuracy 83.8%

# Test Predictions & Accuracy 
tree_prun_03_test  <- mutate(asdf1_test, test_predictions = predict(tree_model_prun_03, newdata = asdf1_test, type = "class"))
mean(~(outcome_type == test_predictions), data = tree_prun_03_test)
# Validation Accuracy 82.7%


# CP - .05
tree_model_prun_05 <- prune(tree_model, cp = 0.05)
rpart.plot(tree_model_prun_05 ,roundint=FALSE,nn=TRUE,extra=4)

# Validation Predictions & Accuracy 
tree_prun_05_validate  <- mutate(asdf1_valid, validation_predictions = predict(tree_model_prun_05, newdata = asdf1_valid, type = "class"))
mean(~(outcome_type == validation_predictions), data = tree_prun_05_validate)
# Validation Accuracy 72.8

# Test Predictions & Accuracy 
tree_prun_05_test  <- mutate(asdf1_test, test_predictions = predict(tree_model_prun_05, newdata = asdf1_test, type = "class"))
mean(~(outcome_type == test_predictions), data = tree_prun_05_test)
# Validation Accuracy 74.4%


# CP - .07
tree_model_prun_07 <- prune(tree_model, cp = 0.07)
rpart.plot(tree_model_prun_07,roundint=FALSE,nn=TRUE,extra=4)

# Validation Predictions & Accuracy 
tree_prun_07_validate  <- mutate(asdf1_valid, validation_predictions = predict(tree_model_prun_07, newdata = asdf1_valid, type = "class"))
mean(~(outcome_type == validation_predictions), data = tree_prun_07_validate)
# Validation Accuracy 74.4%

# Test Predictions & Accuracy 
tree_prun_07_test  <- mutate(asdf1_test, test_predictions = predict(tree_model_prun_07, newdata = asdf1_test, type = "class"))
mean(~(outcome_type == test_predictions), data = tree_prun_07_test)
# Validation Accuracy 82.7%









# Decision Tree - KFold    

# Model - 10 Complexity Parameters, 10 Folds 
tree <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = "rpart", trControl = trainControl("cv", number = 10 ), tuneLength = 10)     

# Model Summary
tree
tree$results
plot(tree)
# Best CP = 0.001798561

# Model w/ Recommended CP
tree_model_kfold <- rpart(outcome_type ~ ., data = asdf1_kfold_train, method="class" , cp = 0.005338078)
rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 1)
rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 4)

# Model Predictions & Accuracy 
asdf1_tree_kfold_predictions <- mutate(asdf1_kfold_test, model_predictions = predict(tree_model_kfold, newdata = asdf1_kfold_test, type = "class"))
mean(~(outcome_type == model_predictions), data = asdf1_tree_kfold_predictions )
# Model Accuracy X%


#--------------------------------------------------------------------------#

# KNN - TVT

# KNN - KFold


#--------------------------------------------------------------------------#

# NNet - TVT 

# Model
nn1 <- nnet(outcome_type ~ ., data = asdf1_norm_train, size = 1, linout = F, decay = 0.05)

# Validation Predictions
nn_predictions <- mutate(asdf1_norm_valid, validation_predictions = predict(nn1,asdf1_norm_valid) %>% round())

# Validation Accuracy
mean(~(outcome_type == validation_predictions),data = nn_predictions) 
# Validation Accuracy - X%


# Validation Predictions
nn_predictions <- mutate(asdf1_norm_test, test_predictions = predict(nn1,asdf1_norm_test) %>% round())

# Validation Accuracy
mean(~(outcome_type == test_predictions),data = nn_predictions) 
# Validation Accuracy - 78.01%






# NNet - KFold
# Model - 10 Folds 
nnet_kfold <- train(outcome_type ~ . , data = asdf1_norm_kfold_train ,method = 'avNNet', trControl = trainControl("cv", number = 1 ))     

# Model Summary
nnet_kfold

# Model Predictions & Accuracy 
asdf1_nnet_kfold_predictions <- mutate(asdf1_norm_kfold_test, model_predictions = predict(nnet_kfold, newdata = asdf1_norm_kfold_test))
mean(~(outcome_type == model_predictions), data = asdf1_nnet_kfold_predictions )
# Model Accuracy X%

#--------------------------------------------------------------------------#

# SVM - TVT 

# Model - Cost 20
svmfit <- svm(outcome_type ~ ., data = asdf1_norm_train , type = "C-classification", cost = 20, scale = FALSE, probability = TRUE)

# Validation Predictions
svm_validation_predictions <- predict(svmfit,asdf1_norm_valid,probability=TRUE)
svm_validation_predictions <- mutate(asdf1_norm_valid, validation_predictions = attributes(svm_validation_predictions)$probabilities[,1]) #DF with Predictions
svm_validation_predictions <- mutate(svm_validation_predictions, validation_predictions = round(validation_predictions)) # Rounding

# Validation Accuracy
mean(~(outcome_type == validation_predictions), data = svm_validation_predictions)
# Validation Accuracy - 82.5%%


# Test Predictions
svm_test_predictions <- predict(svmfit,asdf1_norm_test,probability=TRUE)
svm_test_predictions <- mutate(asdf1_norm_test, test_predictions = attributes(svm_test_predictions)$probabilities[,1]) #DF with Predictions
svm_test_predictions <- mutate(svm_test_predictions, test_predictions = round(test_predictions)) # Rounding

# Test Accuracy
mean(~(outcome_type == test_predictions), data = svm_test_predictions)
# Validation Accuracy - 81.1%%


# SVM - KFold 
# Model - 10 Folds 
svm_kfold <- train(outcome_type ~ . , data = asdf1_norm_kfold_train ,method = 'svmLinearWeights2', trControl = trainControl("cv", number = 1 ))     


# Model Summary
svm_kfold

# Model Predictions & Accuracy 
asdf1_nnet_kfold_predictions <- mutate(asdf1_norm_kfold_test, model_predictions = predict(nnet_kfold, newdata = asdf1_norm_kfold_test))
mean(~(outcome_type == model_predictions), data = asdf1_nnet_kfold_predictions )
# Model Accuracy X%

