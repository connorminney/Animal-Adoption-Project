# Animal Shelter Outcommes
# Developed by Connor Minney & Kaleb Tucker

# Load Packages
  pacman::p_load(dplyr, stringr, rpart, rpart.plot, mosaic, FNN,fastDummies, e1071, nnet, corrplot, caret, arm, brnn, LiblineaR, sampling)
  options(scipen = 10)
  
# Import Data
  asdf <- read.csv("C:/Users/kaleb/OneDrive/Documents/BC/Spring 2021/Machine Learning/Group Project/AAC Animal Intakes.csv")

  
# Resampling
  set.seed(42)
  asdf1 <- strata(asdf, stratanames=NULL, 2000, method="srswor", pik,description=FALSE)
  asdf1 <- getdata(asdf,asdf1)
  
  # Segregate Adoptions & Euthanasia into separate dataframes
  # adoptions <- asdf[asdf$outcome_type == "Adoption" | asdf$outcome_type == "Return to Owner", ]
  # euthanasia <- asdf[asdf$outcome_type != "Adoption", ]
  # euthanasia <- euthanasia[euthanasia$outcome_type != "Return to Owner", ]
  
  # Create Random Sample of Rows
  # adoptionrows <- sample(nrow(adoptions))
  # euthanasiarows <- sample(nrow(euthanasia))  
   
  # Reduce Adoptions & Euthanasia Dataframes
  # adoptions <- adoptions[adoptionrows, ]
  # adoptions <- adoptions[0:12146, ] # 48,385 total records, reduce to 12,146 (~60%) to create 20,000 total records. 
  # euthanasia <- euthanasia[euthanasiarows, ] 
  # euthanasia <- euthanasia[0:7854, ] #31,287 total records, reduce to 7,854 to create 20,000 total records. 
  
  # asdf1 <- rbind(adoptions, euthanasia)
  # remove(adoptions)
  # remove(euthanasia)
  
# Data Pre-Processing
    
  # Remove any record where fields are incomplete
    asdf1 <- na.omit(asdf1)
  
  # Remove Unnecessary Variables
    asdf1 <- dplyr::select(asdf1, -date_of_birth, -animal_id_intake, -animal_id_outcome, -outcome_subtype, -age_upon_outcome, -age_upon_outcome_.years., 
                   -age_upon_outcome_age_group, -outcome_datetime, -outcome_year, -outcome_monthyear, -outcome_number,
                   -dob_year, -dob_month, -dob_monthyear, -age_upon_intake, -found_location, -count, -age_upon_intake_.years., -age_upon_intake_age_group,
                   -intake_datetime, -intake_year, -intake_monthyear, -intake_number, -time_in_shelter)
  
  # Binary Outcome Type - Adopted or Other
    asdf1 <- mutate(asdf1, outcome_type = ifelse(str_detect(outcome_type, "Adopt") == TRUE, "Adoption", ifelse(str_detect(outcome_type, "Return") == TRUE, "Adoption", "Euthanasia")))
    
  # Sex Upon Intake
    asdf1 <- mutate(asdf1, 
                   sex_upon_intake1 = ifelse(str_detect(sex_upon_intake, "male") == TRUE, "F","M"),
                   sex_upon_intake2 = ifelse(str_detect(sex_upon_intake, "Intact") == TRUE, "Intact",
                                            ifelse(str_detect(sex_upon_intake, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf1 <- dplyr::select(asdf1, -sex_upon_intake) 
    
    asdf1 <- mutate(asdf1, 
                   sex_upon_outcome1 = ifelse(str_detect(sex_upon_outcome, "male") == TRUE, "F","M"),
                   sex_upon_outcome2 = ifelse(str_detect(sex_upon_outcome, "Intact") == TRUE, "Intact",
                                             ifelse(str_detect(sex_upon_outcome, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf1 <- dplyr::select(asdf1, -sex_upon_outcome)
    
  # Drop Cats, Birds & Other Animals
    asdf1 <- asdf1[asdf1$animal_type == "Dog", ]
    asdf1 <- dplyr::select(asdf1, -animal_type)
    
  # Breeds
    
    # Create a flag to determine if the animal is a "mix" or not (i.e., Rottweiler vs. Rottweiler Mix)  
    asdf1 <- dplyr::mutate(asdf1, Mix = ifelse(stringr::str_detect(breed, "Mix") == TRUE, "T","F"))
    
    # Remove Mix and Second Breed if indicated by "/" (i.e., Labrador Retriever / Pit Bull -> Labrador Retriever w/ Mix Flag)
    asdf1$breed <- asdf1$breed %>% str_replace("Mix", "")
    asdf1$breed <- asdf1$breed %>% str_replace("/.+", "")
    

  # Factor Categorical Variables
    Categorical_Variables <- c("outcome_type", "outcome_month","outcome_weekday","outcome_hour","breed","color",
                               "intake_condition","intake_type","intake_weekday","intake_hour","sex_upon_intake1","sex_upon_intake2",
                               "sex_upon_outcome1","sex_upon_outcome2","Mix")
    asdf1[,Categorical_Variables] <- lapply(asdf1[,Categorical_Variables], as.factor)
    summary(asdf1)
    
    
  # Check Correlations
    # As most of these items are qualitative, we are unable to determine correlations w/o extending the dataset so sufficiently large
    # that it would be impracticle to perform (e.g., 1700 columns for dog breeds alone)
    # asdf1_corr <- dummy_cols(asdf1, select_columns = c("outcome_type", "age_upon_outcome_.days.", "outcome_month", "outcome_weekday", "outcome_hour", "breed","color","intake_condition","intake_type","age_upon_intake_.days." , "intake_weekday", "intake_hour","time_in_shelter_days","sex_upon_intake1","sex_upon_intake2"), remove_first_dummy = TRUE)


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
                             age_upon_outcome_.days. = (age_upon_outcome_.days.-min(age_upon_outcome_.days.))/(max(age_upon_outcome_.days.)-min(age_upon_outcome_.days.)),
                             age_upon_intake_.days. = (age_upon_intake_.days.-min(age_upon_intake_.days.))/(max(age_upon_intake_.days.)-min(age_upon_intake_.days.)),
                             time_in_shelter_days = (time_in_shelter_days - min(time_in_shelter_days))/(max(time_in_shelter_days)-min(time_in_shelter_days)))
    
    asdf1_norm_train <- filter(asdf1_data_norm, tvt == 'train') %>% dplyr::select(-tvt, -kfold)
    asdf1_norm_valid <- filter(asdf1_data_norm, tvt == 'valid')  %>% dplyr::select(-tvt, -kfold)
    asdf1_norm_test <- filter(asdf1_data_norm, tvt == 'test')  %>% dplyr::select(-tvt, -kfold)
    
    asdf1_norm_kfold_train <- filter(asdf1_data_norm, kfold == 'train') %>% dplyr::select(-kfold, -tvt)
    asdf1_norm_kfold_test <- filter(asdf1_data_norm, kfold == 'test')  %>% dplyr::select(-kfold, -tvt)
    
    
################################################################################    
# Build the Models

    # Logistic Regression - TVT
    # Notes - w/ Logistic Regressions, we were unable to apply the model to the Validation or Test Data as there were new factors in 
    # validation & test which threw the regression. KFold more appropriate at this time. 
        logistic_model <- glm(outcome_type ~ ., data = asdf1_train, family = binomial)
        summary(logistic_model)
        
        # Validation Predictions
        logistic_predictions <- mutate(asdf1_valid, validation_predictions = predict(logistic_model, asdf1_valid, type="response") %>% round())
        
        # Validation Accuracy
        mean(~(outcome_type == validation_predictions), data = logistic_predictions)
        # Validation Accuracy - XX.X%
        
        
    # Logistic Regression - KFold
      # Model - 10 Folds 
      logistic_kfold <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = 'bayesglm', trControl = trainControl("cv", number = 10 ))     
        
      # Model Summary
      logistic_kfold
    
      # Model Predictions & Accuracy 
      asdf1_logistic_kfold_predictions <- mutate(asdf1_kfold_test, model_predictions = predict(logistic_kfold, newdata = asdf1_kfold_test))
      mean(~(outcome_type == model_predictions), data = asdf1_logistic_kfold_predictions )
      # Model Accuracy 76.1% 
      
      
    #--------------------------------------------------------------------------#
    
    # Stepwise Regression - TVT 
        nullmodel <- glm(outcome_type ~ 1, data = asdf1_train, family = binomial)
        stepmodel <- step(nullmodel, scope=formula(logistic_model))
        summary(stepmodel)
        
        # Validation Predictions & Accuracy
        stepwise_validation  <- mutate(asdf1_valid, validation_predictions = round(predict(stepmodel, newdata = asdf1_valid, type = "response"),0))
        stepwise_validation  <- mutate(stepwise_validation, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption"))
        mean(~(outcome_type == validation_predictions), data = stepwise_validation)
        # Validation Data Accuracy - 80.9%
        
        # Dimension Reduction - glm(formula = outcome_type ~ sex_upon_outcome2 + outcome_hour + intake_type + outcome_weekday 
                                    # + intake_condition + outcome_month + time_in_shelter_days + sex_upon_intake1 + age_upon_intake_.days. 
                                    # + intake_hour + intake_weekday + sex_upon_intake2 + Mix) 
    
    # Stepwise Regression - KFold
        
        # Model - 10 Folds 
        stepwise_kfold <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = 'glmStepAIC', trControl = trainControl("cv", number = 10 ))     
        
        # Model Summary
        Stepwise_kfold
        
    
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
        # Validation Accuracy 82.89%
        
        # CP - .03
        tree_model_prun_03 <- prune(tree_model, cp = 0.03)
        rpart.plot(tree_model_prun_03,roundint=FALSE,nn=TRUE,extra=4)
        
        # Validation Predictions & Accuracy 
        tree_prun_03_validate  <- mutate(asdf1_valid, validation_predictions = predict(tree_model_prun_03, newdata = asdf1_valid, type = "class"))
        mean(~(outcome_type == validation_predictions), data = tree_prun_03_validate)
        # Validation Accuracy 80.29%
        
        
    
    # Decision Tree - KFold    
        
        # Model - 8 Complexity Parameters, 10 Folds 
        tree <- train(outcome_type ~ . , data = asdf1_kfold_train ,method = "rpart", trControl = trainControl("cv", number = 10 ), tuneLength = 8)     
    
        # Model Summary
        tree
        tree$results
        plot(tree)
        # Best CP = 0.01141553
        
        # Model w/ Recommended CP
        tree_model_kfold <- rpart(outcome_type ~ ., data = asdf1_kfold_train, method="class" , cp = 0.01141553)
        rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 1)
        rpart.plot(tree_model_kfold, roundint = FALSE, nn = TRUE, extra = 4)
        
        # Model Predictions & Accuracy 
        asdf1_tree_kfold_predictions <- mutate(asdf1_kfold_test, model_predictions = predict(tree_model_kfold, newdata = asdf1_kfold_test, type = "class"))
        mean(~(outcome_type == model_predictions), data = asdf1_tree_kfold_predictions )
        # Model Accuracy 80.09% 
        
        
    #--------------------------------------------------------------------------#
    
    # KNN - TVT
    
    # KNN - KFold
    
        
    #--------------------------------------------------------------------------#
    
    # NNet - TVT 
      
      # Model
      nn1 <- nnet(outcome_type ~ ., data = asdf1_norm_train, size = 1, linout = F, decay = 0.05)
      
      # Validation Predictions
      nn_predictions <- mutate(asdf1_norm_valid, validation_predictions = predict(nn1,asdf1_norm_valid) %>% round())
      nn_predictions  <- mutate(nn_predictions, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption"))
      
      # Validation Accuracy
      mean(~(outcome_type == validation_predictions),data = nn_predictions) 
      # Validation Accuracy - 80.6%
    
    # NNet - KFold
      # Model - 10 Folds 
      nnet_kfold <- train(outcome_type ~ . , data = asdf1_norm_kfold_train ,method = 'avNNet', trControl = trainControl("cv", number = 1 ))     
      
      # Model Summary
      nnet_kfold
      
      # Model Predictions & Accuracy 
      asdf1_nnet_kfold_predictions <- mutate(asdf1_norm_kfold_test, model_predictions = predict(nnet_kfold, newdata = asdf1_norm_kfold_test))
      mean(~(outcome_type == model_predictions), data = asdf1_nnet_kfold_predictions )
      # Model Accuracy 80.84% 
      
    #--------------------------------------------------------------------------#
        
    # SVM - TVT 
        
        # Model - Cost 20
        svmfit <- svm(outcome_type ~ ., data = asdf1_norm_train , type = "C-classification", cost = 20, scale = FALSE, probability = TRUE)
         
        # Validation Predictions
        svm_validation_predictions <- predict(svmfit,asdf1_norm_valid,probability=TRUE)
        svm_validation_predictions <- mutate(asdf1_norm_valid, train_pred_svm_prob = attributes(svm_validation_predictions)$probabilities[,1]) #DF with Predictions
        svm_validation_predictions <- mutate(svm_validation_predictions,train_pred_svm_rounded = round(train_pred_svm_prob)) # Rounding
        svm_validation_predictions  <- mutate(svm_validation_predictions, validation_predictions = ifelse(train_pred_svm_rounded == 1, "Adoption","Euthanasia")) # Assign Euthanasia v. Adoption
        
        # Validation Accuracy
        mean(~(outcome_type == validation_predictions), data = svm_validation_predictions)
        # Validation Accuracy - 79.1%
        
        
    # SVM - KFold (working through now..pick back up)
        # Model - 10 Folds 
        svm_kfold <- train(outcome_type ~ . , data = asdf1_norm_kfold_train ,method = 'svmLinearWeights2', trControl = trainControl("cv", number = 1 ))     
        
        
        # Model Summary
        svm_kfold
        
        # Model Predictions & Accuracy 
        asdf1_nnet_kfold_predictions <- mutate(asdf1_norm_kfold_test, model_predictions = predict(nnet_kfold, newdata = asdf1_norm_kfold_test))
        mean(~(outcome_type == model_predictions), data = asdf1_nnet_kfold_predictions )
        # Model Accuracy XX.XX% 