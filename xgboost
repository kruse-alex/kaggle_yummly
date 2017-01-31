#############################################################################################################################################
# DATA LOADING
#############################################################################################################################################

# activate package
library(jsonlite) # load data
library(tm) # save in document matrix cia corpus
library(data.table)
library(Matrix)
library(caret)
library(SnowballC) # for word stemming
library(xgboost) # classification algorithm
library(Ckmeans.1d.dp)

# load data files and flatten
train_raw  <- fromJSON("train.json", flatten = TRUE) # flatten option removes nestet data frame
submit_raw <- fromJSON("test.json", flatten = TRUE)

#############################################################################################################################################
# DATA PROCESSING
#############################################################################################################################################

# preprocess the ingredients (basic)
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=tolower) # strings to lowercase
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("-", "_", x)) # replace strings
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) # allow regular character and spaces

submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=tolower) # strings to lowercase
submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=function(x) gsub("-", "_", x)) # replace strings
submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) # allow regular character and spaces

# create a matrix of ingredients in both the TRAIN and SUBMIT set
c_ingredients <- c(Corpus(VectorSource(train_raw$ingredients)), Corpus(VectorSource(submit_raw$ingredients))) # corpus from tm package for text mining.

# preprocess the ingredients (advanced)
c_ingredients <- tm_map(c_ingredients, stemDocument)

# create simple DTM (saves documents (ingredients) in rows and columns represents each ingrdient)
c_ingredientsDTM <- DocumentTermMatrix(c_ingredients) 
c_ingredientsDTM <- removeSparseTerms(c_ingredientsDTM, 1-3/nrow(c_ingredientsDTM)) # remove if < 3 occurances
c_ingredientsDTM <- as.data.frame(as.matrix(c_ingredientsDTM))

# feature engineering
c_ingredientsDTM$ingredients_count  <- rowSums(c_ingredientsDTM) # simple count of ingredients per receipe

# add cuisine for TRAIN set, default to "italian" for the SUBMIT set
c_ingredientsDTM$cuisine <- as.factor(c(train_raw$cuisine, rep("italian", nrow(submit_raw))))

# split the DTM into TRAIN and SUBMIT sets
dtm_train  <- c_ingredientsDTM[1:nrow(train_raw), ]
dtm_submit <- c_ingredientsDTM[-(1:nrow(train_raw)), ]

#############################################################################################################################################
# MODELING
#############################################################################################################################################

# prepare the spare matrix (note: feature index in xgboost starts from 0)
xgbmat <- xgb.DMatrix(Matrix(data.matrix(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])), label=as.numeric(dtm_train$cuisine)-1)

# train our multiclass classification model using softmax
xgb <- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)

# predict on the SUBMIT set and change cuisine back to string
xgb.submit      <- predict(xgb, newdata = data.matrix(dtm_submit[, !colnames(dtm_submit) %in% c("cuisine")]))
xgb.submit.text <- levels(dtm_train$cuisine)[xgb.submit+1]

#############################################################################################################################################
# CREATE SUBMISSION FILE
#############################################################################################################################################

# load sample submission file to use as a template
sample_sub <- read.csv("sample_submission.csv")

# build and write the submission file
submit_match   <- cbind(as.data.frame(submit_raw$id), as.data.frame(xgb.submit.text))
colnames(submit_match) <- c("id", "cuisine")
write.csv(submit_match, file = 'xgboost_multiclass.csv', row.names=F, quote=F)

# plot the most important features
names <- colnames(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30,])
