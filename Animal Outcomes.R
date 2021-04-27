# Animal Shelter Outcommes
# Developed by Connor Minney & Kaleb Tucker

# Load Packages
  pacman::p_load(dplyr, stringr, rpart, rpart.plot, mosaic, FNN,fastDummies, e1071, nnet, corrplot, caret, arm)
  options(scipen = 10)
  
# Import Data
  asdf <- read.csv("C:/Users/kaleb/OneDrive/Documents/BC/Spring 2021/Machine Learning/Group Project/AAC Animal Intakes.csv")

# Data Pre-Processing
    
  # Remove any record where fields are incomplete
    asdf <- na.omit(asdf)
  
  # Remove Unnecessary Variables
    asdf <- dplyr::select(asdf, -date_of_birth, -animal_id_intake, -animal_id_outcome, -outcome_subtype, -age_upon_outcome, -age_upon_outcome_.years., 
                   -age_upon_outcome_age_group, -outcome_datetime, -outcome_year, -outcome_monthyear, -outcome_number,
                   -dob_year, -dob_month, -dob_monthyear, -age_upon_intake, -found_location, -count, -age_upon_intake_.years., -age_upon_intake_age_group,
                   -intake_datetime, -intake_year, -intake_monthyear, -intake_number, -time_in_shelter)
  
  # Binary Outcome Type - Adopted or Other
    asdf <- mutate(asdf, outcome_type = ifelse(str_detect(outcome_type, "Adopt") == TRUE, "Adoption", ifelse(str_detect(outcome_type, "Return") == TRUE, "Adoption", "Euthanasia")))
    
  # Sex Upon Intake
    asdf <- mutate(asdf, 
                   sex_upon_intake1 = ifelse(str_detect(sex_upon_intake, "male") == TRUE, "F","M"),
                   sex_upon_intake2 = ifelse(str_detect(sex_upon_intake, "Intact") == TRUE, "Intact",
                                            ifelse(str_detect(sex_upon_intake, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf <- dplyr::select(asdf, -sex_upon_intake) 
    
    asdf <- mutate(asdf, 
                   sex_upon_outcome1 = ifelse(str_detect(sex_upon_outcome, "male") == TRUE, "F","M"),
                   sex_upon_outcome2 = ifelse(str_detect(sex_upon_outcome, "Intact") == TRUE, "Intact",
                                             ifelse(str_detect(sex_upon_outcome, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf <- dplyr::select(asdf, -sex_upon_outcome)
    
  # Drop Cats, Birds & Other Animals
    asdf <- asdf[asdf$animal_type == "Dog", ]
    asdf <- dplyr::select(asdf, -animal_type)
    
  # Breeds
    
    # Create a flag to determine if the animal is a "mix" or not (i.e., Rottweiler vs. Rottweiler Mix)  
    asdf <- dplyr::mutate(asdf, Mix = ifelse(stringr::str_detect(breed, "Mix") == TRUE, "T","F"))
    
    # Remove Mix and Second Breed if indicated by "/" (i.e., Labrador Retriever / Pit Bull -> Labrador Retriever w/ Mix Flag)
    asdf$breed <- asdf$breed %>% str_replace("Mix", "")
    asdf$breed <- asdf$breed %>% str_replace("/.+", "")
    

  # Factor Categorical Variables
    Categorical_Variables <- c("outcome_type", "outcome_month","outcome_weekday","outcome_hour","breed","color",
                               "intake_condition","intake_type","intake_weekday","intake_hour","sex_upon_intake1","sex_upon_intake2",
                               "sex_upon_outcome1","sex_upon_outcome2","Mix")
    asdf[,Categorical_Variables] <- lapply(asdf[,Categorical_Variables], as.factor)
    summary(asdf)
    
    
  # Check Correlations
    # As most of these items are qualitative, we are unable to determine correlations w/o extending the dataset so sufficiently large
    # that it would be impracticle to perform (e.g., 1700 columns for dog breeds alone)
    # asdf_corr <- dummy_cols(asdf, select_columns = c("outcome_type", "age_upon_outcome_.days.", "outcome_month", "outcome_weekday", "outcome_hour", "breed","color","intake_condition","intake_type","age_upon_intake_.days." , "intake_weekday", "intake_hour","time_in_shelter_days","sex_upon_intake1","sex_upon_intake2"), remove_first_dummy = TRUE)


  # Splitting Data 
    # Split the Data into Training, Validation, and Testing Sets for TVT Validations
      set.seed(1)
      tvt <- sample(c('train','valid','test'),size = nrow(asdf),replace = TRUE, prob = c(0.6,0.3,0.1))
      asdf <- mutate(asdf,tvt)
      
      asdf_train <- filter(asdf, tvt == 'train') %>% dplyr::select(-tvt)
      asdf_valid <- filter(asdf, tvt == 'valid')  %>% dplyr::select(-tvt)
      asdf_test <- filter(asdf, tvt == 'test')  %>% dplyr::select(-tvt)
      
    # Split the Data into Training and Testing for KFold Validations
      kfold <- sample(c('train', 'test'),size = nrow(asdf),replace = TRUE, prob = c(0.8,0.2))
      asdf <- mutate(asdf,kfold)
      
      asdf_kfold_train <- filter(asdf, kfold == 'train') %>% dplyr::select(-kfold, -tvt)
      asdf_kfold_test <- filter(asdf, kfold == 'test')  %>% dplyr::select(-kfold, -tvt)
    
    
  # Normalization for KNN & SVM
    asdf_data_norm <- mutate(asdf, 
                             age_upon_outcome_.days. = (age_upon_outcome_.days.-min(age_upon_outcome_.days.))/(max(age_upon_outcome_.days.)-min(age_upon_outcome_.days.)),
                             age_upon_intake_.days. = (age_upon_intake_.days.-min(age_upon_intake_.days.))/(max(age_upon_intake_.days.)-min(age_upon_intake_.days.)),
                             time_in_shelter_days = (time_in_shelter_days - min(time_in_shelter_days))/(max(time_in_shelter_days)-min(time_in_shelter_days)))
    
    asdf_norm_train <- filter(asdf_data_norm, tvt == 'train') %>% dplyr::select(-tvt, -kfold)
    asdf_norm_valid <- filter(asdf_data_norm, tvt == 'valid')  %>% dplyr::select(-tvt, -kfold)
    asdf_norm_test <- filter(asdf_data_norm, tvt == 'test')  %>% dplyr::select(-tvt, -kfold)
    
    asdf_norm_kfold_train <- filter(asdf_data_norm, kfold == 'train') %>% dplyr::select(-kfold, -tvt)
    asdf_norm_kfold_test <- filter(asdf_data_norm, kfold == 'test')  %>% dplyr::select(-kfold, -tvt)
    
    
################################################################################    
# Build the Models

    # Logistic Regression - TVT
        logistic_model <- glm(outcome_type ~ ., data = asdf_train, family = binomial)
        summary(logistic_model)
        
        # Validation Predictions
        logistic_predictions <- mutate(asdf_valid, validation_predictions = predict(logistic_model, asdf_valid, type="response") %>% round()) #dplyr
        
        # Validation Accuracy
        mean(~(outcome_type == validation_predictions), data = logistic_predictions)
        # Validation Accuracy - XX.X%
        
      ###### Review Note - cannot perform this as there are "new factors" with issues randomly assigning values. 
        
    # Logistic Regression - KFold
      # Model - 10 Folds 
      logistic_kfold <- train(outcome_type ~ . , data = asdf_kfold_train ,method = 'bayesglm', trControl = trainControl("cv", number = 10 ))     
        
      # Model Summary
      logistic_kfold
    
      # Model Predictions & Accuracy 
      asdf_logistic_kfold_predictions <- mutate(asdf_kfold_test, model_predictions = predict(logistic_kfold, newdata = asdf_kfold_test))
      mean(~(outcome_type == model_predictions), data = asdf_logistic_kfold_predictions )
      # Model Accuracy 80.73% 
      
      
    #--------------------------------------------------------------------------#
    
    # Stepwise Regression - TVT 
        nullmodel <- glm(outcome_type ~ 1, data = asdf_train, family = binomial)
        stepmodel <- step(nullmodel, scope=formula(logistic_model))
        summary(stepmodel)
        
        # Validation Predictions & Accuracy
        stepwise_validation  <- mutate(asdf_valid, validation_predictions = round(predict(stepmodel, newdata = asdf_valid, type = "response"),0))
        stepwise_validation  <- mutate(stepwise_validation, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption"))
        mean(~(outcome_type == validation_predictions), data = stepwise_validation)
        # Validation Data Accuracy - 80.9%
        
        # Dimension Reduction - glm(formula = outcome_type ~ sex_upon_outcome2 + outcome_hour + intake_type + outcome_weekday 
                                    # + intake_condition + outcome_month + time_in_shelter_days + sex_upon_intake1 + age_upon_intake_.days. 
                                    # + intake_hour + intake_weekday + sex_upon_intake2 + Mix) 
    
    # Stepwise Regression - KFold
        
        # Model - 10 Folds 
         stepwise_kfold <- train(outcome_type ~ . , data = asdf_kfold_train ,method = 'glmStepAIC', trControl = trainControl("cv", number = 10 ))     
        
        # Model Summary
        logistic_kfold
        
    
    #--------------------------------------------------------------------------#
    
    # Decision Tree - TVT    
        
        # Model
        tree_model <- rpart(outcome_type ~ ., data = asdf_train, method = "class")
        
        # CP - .01
        tree_model_prun_01 <- prune(tree_model, cp = 0.01)
        rpart.plot(tree_model_prun_01,roundint=FALSE,nn=TRUE,extra=4)
        
        # Validation Predictions & Accuracy
        tree_prun_01_validate  <- mutate(asdf_valid, validation_predictions = predict(tree_model_prun_01, newdata = asdf_valid, type = "class"))
        mean(~(outcome_type == validation_predictions), data = tree_prun_01_validate)
        # Validation Accuracy 82.89%
        
        # CP - .03
        tree_model_prun_03 <- prune(tree_model, cp = 0.03)
        rpart.plot(tree_model_prun_03,roundint=FALSE,nn=TRUE,extra=4)
        
        # Validation Predictions & Accuracy 
        tree_prun_03_validate  <- mutate(asdf_valid, validation_predictions = predict(tree_model_prun_03, newdata = asdf_valid, type = "class"))
        mean(~(outcome_type == validation_predictions), data = tree_prun_03_validate)
        # Validation Accuracy 80.29%
        
        
    
    # Decision Tree - KFold    
        
        # Model - 8 Complexity Parameters, 10 Folds 
        tree <- train(outcome_type ~ . , data = asdf_kfold_train ,method = "rpart", trControl = trainControl("cv", number = 10 ), tuneLength = 8)     
    
        # Model Summary
        tree
        tree$results
        plot(tree)
        # Best CP = 0.00414823
        
        # Model w/ Recommended CP
        tree_model_kfold <- rpart(outcome_type ~ ., data = asdf_kfold_train, method="class" , cp = 0.00414823)
        rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 1)
        rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 4)
        
        # Model Predictions & Accuracy 
        asdf_tree_kfold_predictions <- mutate(asdf_kfold_test, model_predictions = predict(tree_model_kfold, newdata = asdf_kfold_test, type = "class"))
        mean(~(outcome_type == model_predictions), data = asdf_tree_kfold_predictions )
        # Model Accuracy 83.43% 
        
        
    #--------------------------------------------------------------------------#
    # KNN - TVT
    # KNN - KFold
    #--------------------------------------------------------------------------#
    
    # NNet - TVT 
      
      # Model
      nn1 <- nnet(outcome_type ~ ., data = asdf_norm_train, size = 1, linout = F, decay = 0.05)
      
      # Validation Predictions
      nn_predictions <- mutate(asdf_norm_valid, validation_predictions = predict(nn1,asdf_norm_valid) %>% round())
      nn_predictions  <- mutate(nn_predictions, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption"))
      
      # Validation Accuracy
      mean(~(outcome_type == validation_predictions),data = nn_predictions) 
      # Validation Accuracy - 80.6%
    
    # NNet - KFold
      
      
    #--------------------------------------------------------------------------#
        
    # SVM - TVT 
        
        # Model - Cost 20
        svmfit <- svm(outcome_type ~ ., data = asdf_norm_train , type = "C-classification", cost = 20, scale = FALSE, probability = TRUE)
         
        # Validation Predictions
        svm_validation_predictions <- predict(svmfit,asdf_norm_valid,probability=TRUE)
        svm_validation_predictions <- mutate(asdf_norm_valid, train_pred_svm_prob = attributes(svm_validation_predictions)$probabilities[,1]) #DF with Predictions
        svm_validation_predictions <- mutate(svm_validation_predictions,train_pred_svm_rounded = round(train_pred_svm_prob)) # Rounding
        svm_validation_predictions  <- mutate(svm_validation_predictions, validation_predictions = ifelse(train_pred_svm_rounded == 1, "Adoption","Euthanasia")) # Assign Euthanasia v. Adoption
        
        # Validation Accuracy
        mean(~(outcome_type == validation_predictions), data = svm_validation_predictions)
        # Validation Accuracy - 79.1%
        
        
    # SVM - KFold