# Animal Shelter Outcommes
# Developed by Connor Minney & Kaleb Tucker

# Load Packages
  pacman::p_load(dplyr, stringr, rpart, rpart.plot, mosaic, FNN,fastDummies, e1071, nnet)
  options(scipen = 10)
  
# Import Data
  asdf <- read.csv("C:/Users/kaleb/OneDrive/Documents/BC/Spring 2021/Machine Learning/Group Project/AAC Animal Intakes.csv")

# Data Pre-Processing
    
  # Remove any record where fields are incomplete
    asdf <- na.omit(asdf)
  
  # Remove Unnecessary Variables
    asdf <- select(asdf, -date_of_birth, -animal_id_intake, -animal_id_outcome, -outcome_subtype, -age_upon_outcome, -age_upon_outcome_.years., 
                   -age_upon_outcome_age_group, -outcome_datetime, -outcome_year, -outcome_monthyear, -outcome_number,
                   -dob_year, -dob_month, -dob_monthyear, -age_upon_intake, -found_location, -count, -age_upon_intake_.years., -age_upon_intake_age_group,
                   -intake_datetime, -intake_month, -intake_year, -intake_monthyear, -intake_number, -time_in_shelter)
  
  # Binary Outcome Type - Adopted or Other
    asdf <- mutate(asdf, outcome_type = ifelse(str_detect(outcome_type, "Adopt") == TRUE, "Adoption", ifelse(str_detect(outcome_type, "Return") == TRUE, "Adoption", "Euthanasia")))
    
  # Sex Upon Intake
    asdf <- mutate(asdf, 
                   sex_upon_intake1 = ifelse(str_detect(sex_upon_intake, "male") == TRUE, "F","M"),
                   sex_upon_intake2 = ifelse(str_detect(sex_upon_intake, "Intact") == TRUE, "Intact",
                                            ifelse(str_detect(sex_upon_intake, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf <- select(asdf, -sex_upon_intake) 
    
    asdf <- mutate(asdf, 
                   sex_upon_outcome1 = ifelse(str_detect(sex_upon_outcome, "male") == TRUE, "F","M"),
                   sex_upon_outcome2 = ifelse(str_detect(sex_upon_outcome, "Intact") == TRUE, "Intact",
                                             ifelse(str_detect(sex_upon_outcome, "Spayed") == TRUE, "Spayed","Neutered")))
    asdf <- select(asdf, -sex_upon_outcome)
    
  # Drop Cats, Birds & Other Animals
    asdf <- asdf[asdf$animal_type == "Dog", ]
    asdf <- select(asdf, -animal_type)
    
  # Breeds
    
    # Create a flag to determine if the animal is a "mix" or not (i.e., Rottweiler vs. Rottweiler Mix)  
    asdf <- mutate(asdf, Mix = ifelse(str_detect(breed, "Mix") == TRUE, "T","F"))
    
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

    
  # Split the Data into Training, Validation, and Testing Sets
    set.seed(1)
    tvt <- sample(c('train','valid','test'),size = nrow(asdf),replace = TRUE, prob = c(0.6,0.3,0.1))
    asdf <- mutate(asdf,tvt)
    
    asdf_train <- filter(asdf, tvt == 'train') %>% select(-tvt)
    asdf_valid <- filter(asdf, tvt == 'valid')  %>% select(-tvt)
    asdf_test <- filter(asdf, tvt == 'test')  %>% select(-tvt)
    
    
    
    
  # Normalization for KNN & SVM
    # Note for Connor - age upon outcome & intake appear to be almost identical. 
    asdf_data_norm <- mutate(asdf, 
                             age_upon_outcome_.days. = (age_upon_outcome_.days.-min(age_upon_outcome_.days.))/(max(age_upon_outcome_.days.)-min(age_upon_outcome_.days.)),
                             age_upon_intake_.days. = (age_upon_intake_.days.-min(age_upon_intake_.days.))/(max(age_upon_intake_.days.)-min(age_upon_intake_.days.)),
                             time_in_shelter_days = (time_in_shelter_days - min(time_in_shelter_days))/(max(time_in_shelter_days)-min(time_in_shelter_days)))
    
    asdf_norm_train <- filter(asdf_data_norm, tvt == 'train') %>% select(-tvt)
    asdf_norm_valid <- filter(asdf_data_norm, tvt == 'valid')  %>% select(-tvt)
    asdf_morm_test <- filter(asdf_data_norm, tvt == 'test')  %>% select(-tvt)
    
    
# Build the Models

    # Logistic Regression
      logistic_model <- glm(outcome_type ~ ., data = asdf_train, family = binomial)
      summary(logistic_model)
      
      # Validation Predictions
      logistic_predictions <- mutate(asdf_valid, validation_predictions = predict(logistic_model, asdf_valid, type="response") %>% round()) #dplyr
      
      # Validation Accuracy
      mean(~(outcome_type == validation_predictions), data = logistic_predictions)
      # Validation Accuracy - XX.X%
      
      
    # Stepwise Regression
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
    
      
    # Decision Tree
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
      
      
    # KNN
      
      
      
    # NNet
      
      # Model
      nn1 <- nnet(outcome_type ~ ., data = asdf_norm_train, size = 1, linout = F, decay = 0.05)
      
      # Validation Predictions
      nn_predictions <- mutate(asdf_norm_valid, validation_predictions = predict(nn1,asdf_norm_valid) %>% round())
      nn_predictions  <- mutate(nn_predictions, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption"))
      
      # Validation Accuracy
      mean(~(outcome_type == validation_predictions),data = nn_predictions) 
      # Validation Accuracy - 80.6%
      
    # Naive Bayes
      
      # NB Data Pre-processing
        # Remove Scale Variables 
        asdf_nb <- select(asdf, - age_upon_outcome_.days., -age_upon_intake_.days., -time_in_shelter_days) 
        
        # Divide the data frame into subsets
        asdf_nb_train <- filter(asdf_nb, tvt =="train") %>% select(-tvt)
        asdf_nb_valid <- filter(asdf_nb, tvt =="valid") %>% select(-tvt)
        asdf_nb_test <- filter(asdf_nb, tvt =="test") %>% select(-tvt)
    
      # NB Model
        nb <- naiveBayes(outcome_type ~ . , data = asdf_nb_train)
        nb
        
        # Validation Predictions
        nb_validation_predictions <- predict(nb, asdf_nb_valid, type = "raw") %>% data.frame()
        nb_validation <- mutate(asdf_nb_valid, validation_predictions = nb_validation_predictions$Euthanasia) # New DF w/ actual v predictions
        nb_validation <- mutate(nb_validation, validation_predictions = round(validation_predictions)) # Round Predictions 
        nb_validation  <- mutate(nb_validation, validation_predictions = ifelse(validation_predictions == 1, "Euthanasia", "Adoption")) # Assign Euthanasia v. Adoption
        
        # Validation Accuracy
        mean(~(outcome_type == validation_predictions), data = nb_validation)
        # Validation Accuracy - 79.9%
        
    # SVM
        
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