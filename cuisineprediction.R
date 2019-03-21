library(jsonlite)
library(tm)
library(data.table)
library(Matrix)
library(caret)
library(SnowballC)
library(xgboost)
library(Ckmeans.1d.dp)

#- load data files
train_raw  <- fromJSON("F:/PROJECTS/Vyakya/train.json", flatten = TRUE)
submit_raw <- fromJSON("F:/PROJECTS/Vyakya/test.json", flatten = TRUE)



#- pre-process the ingredients
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=tolower)
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("-", "_", x)) # allow dash e.g. "low-fat"
train_raw$ingredients <- lapply(train_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) # allow regular character and spaces

submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=tolower)
submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=function(x) gsub("-", "_", x)) # allow dash e.g. "low-fat"
submit_raw$ingredients <- lapply(submit_raw$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) # allow regular character and spaces

#- create a matrix of ingredients in both the TRAIN and SUBMIT set
c_ingredients <- c(VCorpus(VectorSource(train_raw$ingredients)),VCorpus(VectorSource(submit_raw$ingredients)))

c_ingredientsDTM <- DocumentTermMatrix(c_ingredients, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE))) 
c_ingredientsdf <- as.data.frame(as.matrix(c_ingredientsDTM))

#- add cuisine for TRAIN set, default to "italian" for the SUBMIT set
c_ingredientsdf$cuisine <- as.factor(c(train_raw$cuisine, rep("italian", nrow(submit_raw))))

#- split the DTM into TRAIN and SUBMIT sets
dtm_train  <- c_ingredientsdf[1:nrow(train_raw), ]
dtm_submit <- c_ingredientsdf[-(1:nrow(train_raw)), ]

#- nfold CV on TRAIN set
#- [removed]

#- Model 4: xgboost (single model for all cuisine types)
#- prepare the spare matrix (note: feature index in xgboost starts from 0)
xgbmat     <- xgb.DMatrix(Matrix(data.matrix(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])), label=as.numeric(dtm_train$cuisine)-1)

#- train our multiclass classification model using softmax
xgb        <- xgboost(xgbmat, max.depth = 25, eta = 0.3, nround = 200, objective = "multi:softmax", num_class = 20)

#- predict on the SUBMIT set and change cuisine back to string
xgb.submit      <- predict(xgb, newdata = data.matrix(dtm_submit[, !colnames(dtm_submit) %in% c("cuisine")]))
xgb.submit.text <- levels(dtm_train$cuisine)[xgb.submit+1]

#- load sample submission file to use as a template
sample_sub <- read.csv('F:/PROJECTS/Vyakya/sample_submission.csv')

#- build and write the submission file
submit_match   <- cbind(as.data.frame(submit_raw$id), as.data.frame(xgb.submit.text))
colnames(submit_match) <- c("id", "cuisine")
submit_match   <- data.table(submit_match, key="id")
submit_cuisine <- submit_match[id==sample_sub$id, as.matrix(cuisine)]

submission <- data.frame(id = submit_raw$id, cuisine = submit_match$cuisine)
write.csv(submission, file = 'xboost_multiclass.csv', row.names=F, quote=F)
write.csv(xgb$evaluation_log, file = 'boost_multiclass.csv', row.names=F, quote=F)

# plot the most important features
names <- colnames(dtm_train[, !colnames(dtm_train) %in% c("cuisine")])
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:30,])