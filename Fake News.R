# -----------------------------------------------------------------
# Jovan Trajceski
# Date: 19/04/2021
# Data-set: News (fake/true)
# Code-total-run-time: 20 minutes on Intel Core i7
# -----------------------------------------------------------------

# Clear up data in global environment
rm(list=ls())

# ONLY if required: install packages
install.packages('dplyr')
install.packages('tidyr')
install.packages("glmnet")
install.packages("foreach")
install.packages('iterators')
install.packages("parallel")
install.packages("doParallel")
install.packages("readr")
install.packages("stringr")
install.packages("ggplot2",dependencies = TRUE)
install.packages('tm')
install.packages('textstem')
install.packages('wordcloud2')
install.packages('pROC')
install.packages('ROCR')
install.packages('randomForest')
install.packages('naivebayes')
install.packages('caret')
install.packages('naniar')
install.packages('visdat')
install.packages('utf8')
install.packages('lubridate')
install.packages('klaR')
install.packages("tibble")
install.packages("conflicted")
install.packages("arm", dependencies = TRUE)
install.packages("rpart")
install.packages("gbm")
install.packages("hms", dependencies = TRUE)
install.packages("kableExtra", dependencies = TRUE)
install.packages("broom")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("rpart.plot")
install.packages("e1071")
install.packages("nnet")
install.packages("NeuralNetTools")
install.packages("tidytext", type="binary")
install.packages("RcppEigen")
install.packages("R2WinBUGS")
install.packages("foreign")
install.packages("lme4",dependencies = TRUE)
# For dev version
install.packages("devtools")
devtools::install_github("haozhu233/kableExtra")


# Load Libraries:
library(gbm) 
library(conflicted)
library(tidyr)
library(dplyr)
library(tibble)
library(glmnet)
library(foreach)
library(iterators)
library(parallel)
no_of_cores = detectCores()
library(doParallel)
library(readr)
library(stringr)
library(ggplot2)
library(tm)      
library(textstem) 
library(tidytext)
library(wordcloud2)
library(pROC)
library(ROCR)
library(randomForest) 
library(naivebayes)
library(caret)
library(naniar)
library(visdat)
library(utf8)
cl = makeCluster(4)
registerDoParallel(cl)
library(lubridate)
library(klaR)
library(rpart)
library(broom)
library(ggplot2)
library(SnowballC)
library(wordcloud)
library(rpart.plot)
library(e1071)
library(nnet)
library(NeuralNetTools)
library(stringr)
library(arm)
library(knitr)
library(kableExtra)


# ---------------------------------------------------------------
# Step 0: Resolve conflict between packages (please run this!)
# ---------------------------------------------------------------
filter <- dplyr::filter
select <- dplyr::select

# -----------------------
# Step 1: Load datasets
# -----------------------
fake = read_csv("Fake.csv")
true = read_csv("True.csv")

# -------------------------
# Step 2: Review datasets
# -------------------------
head(fake)
head(true)

# ---------------------------------------------------------------------------------------------------
# Step 3: Add column called "category" to datasets to distinguish between fake (0) and real news (1)
# ---------------------------------------------------------------------------------------------------
fake$category <- 0
true$category <- 1

glimpse(fake)
glimpse(true)

# -------------------------------------------------------------------------------------
# Step 3A: Top Words for Fake News by "Title" (only words over 5 characters allowed)
# -------------------------------------------------------------------------------------
mytext_fake = tibble(text = fake$title) %>% 
    unnest_tokens(word, text) %>% 
    group_by(word) %>% 
    count(word, sort = TRUE) %>% mutate(len=nchar(word)) %>% filter(len>5)


wc_fake = ggplot(head(mytext_fake,10), aes(x=reorder(word, -n),y=n)) + 
    geom_col(fill="red") + 
    geom_text(aes(label=n),position=position_dodge(width=0.9), vjust=-0.25) +
    theme_light() + 
    ylab("Number of posts") + 
    xlab("Word") + 
    ggtitle("Top words - Fake News only")


wc_fake

# ------------------------------------------------------------------------------------------
# Step 3B: Top-Words for True/Real News by "Title" (only words over 5 characters allowed)
# -----------------------------------------------------------------------------------------
mytext_true = tibble(text = true$title) %>% 
    unnest_tokens(word, text) %>% 
    group_by(word) %>% 
    count(word, sort = TRUE) %>% mutate(len=nchar(word)) %>% filter(len>5)


wc_true = ggplot(head(mytext_true,10), aes(x=reorder(word, -n),y=n)) + 
    geom_col(fill="green") + 
    geom_text(aes(label=n),position=position_dodge(width=0.9), vjust=-0.25) +
    theme_light() + 
    ylab("Number of posts") + 
    xlab("Word") + 
    ggtitle("Top words - True news only")


wc_true

# ------------------------------------
# Step 4: Merge two datasets into one
# ------------------------------------
news <- bind_rows(fake, true)

# -------------------------------------------------------------------------------------
# Step 5: Clean up date column to create a plot (no need for this to remain in step 4)
# -------------------------------------------------------------------------------------
dataXYZ = bind_rows(true,fake) %>% 
    mutate(formatted_date = mdy(date))

# -------------------------------------------------------------
# Step 5A: Create a plot of fake and true categories over time
# -------------------------------------------------------------
dataXYZ$category <- as.factor(dataXYZ$category)
ggplot(dataXYZ, aes(x=formatted_date,  fill = category)) + 
    geom_density() +
    theme_classic() +
    theme(axis.title = element_text(face = 'bold', size = 15),
          axis.text = element_text(size = 13)) +
    ggtitle("Fake (0) and True (1) news categories over time")+
    theme(legend.position = 'right')

### Note: ### Fake news are more frequent in 2016, 2017 and first half of 2018. In Q4 2018 fake and true news are balanced ###

# --------------------------------
# Step 6: Review merged dataset
# --------------------------------
news %>%
    sample_n(10)

glimpse(news)

# ------------------------------------------------------------
# Step 7: Lets see if the fake and real news are balanced?
# ------------------------------------------------------------
news$category <- as.factor(news$category)

ggplot(news, aes(x = category, fill = category)) + 
    geom_bar() +
    theme_classic() +
    theme(axis.title = element_text(face = 'bold', size = 15),
          axis.text = element_text(size = 13)) +
    ggtitle("Lets see if fake and true news datasets are balanced?")+
    theme(legend.position = 'none')

### NOTE: #### The merged dataset is Balanced - this will make it easier for prediction! ###

# -------------------------
# Step 8: Summarize data
# -------------------------
summary(news)

# -----------------------------------
# Step 9: Check for missing values
# -----------------------------------
summary(is.na(news)) # basic look-up for missing values
vis_miss(news) # 0.3% are missing in the Text column
gg_miss_var(news) # 631 items are missing
news <- na.omit(news) # Remove missing rows
vis_miss(news) # no missing values anymore

# ----------------------------------------------------------
# Step 10: Change data type of "subject" column to a factor
# ----------------------------------------------------------
news$subject <- as.factor(news$subject)

# -------------------------------------------------------
# Step 11: News count by each Subject and create a plot
# -------------------------------------------------------
news %>%
    group_by(subject) %>%
    count() %>%
    arrange(desc(n))

news %>%
    group_by(subject) %>%
    count(sort = TRUE) %>%
    rename(freq = n) %>%
    ggplot(aes(x = reorder(subject, -freq), y = freq)) + 
    geom_bar(stat = 'identity', fill = 'lightgreen') +
    theme_classic() +
    xlab('Subject') +
    ylab('frequency') +
    geom_text(aes(label = freq), vjust = 1.2, fontface = 'bold') +
    ggtitle("Combined News split by Subject")+
    theme(axis.title = element_text(face = 'bold', size = 15),
          axis.text = element_text(size = 13, angle = 90))

### Note: ### "politicsNews" is the Subject with highest frequency (11272) ###

# --------------------------------------
# Step 12: Category & Subject Bar Plot
# --------------------------------------
ggplot(news, aes(x = subject, fill = category)) +
    geom_bar(position = 'dodge', alpha = 0.7) +
    theme_classic() +
    ggtitle("Combined News split by Subject and Category (false and true)")+
    theme(axis.title = element_text(face = 'bold', size = 15),
          axis.text = element_text(size = 13, angle = 90))

### Note: ### It is clear that the true/real news only have 2 subjects: politicsNews and WorldNews ###

# ------------------------------------------------------------
# Step 13: Combine "Title" and "Text" column into one column
# ------------------------------------------------------------
# I have decided to add the title inside the "text" column for better identification and we dont lose that data
news_combo <- news %>% 
    select(title, text, category) %>%
    unite(col = text ,title, text, sep = ' ')  %>%  # Combine 'title' & 'text' column
    mutate(ID = as.character(1:nrow(news)))    # Unique row ID

glimpse(news_combo)

# -------------------------------------------------------------------
# Step 14: Create a corpus (type of object expected by "tm" library) 
# -------------------------------------------------------------------
# One of the nicest features of "tm" package is the variety of bundled transformations to be applied on corpuses.
# to see the list of available transformation methods, simply call the function "getTransformations()"
doc <- VCorpus(VectorSource(news_combo$text))

# ------------------------------------------------------------------------------------------------------
# Step 15: Pre-process Text (remove: lower case, numbers, punctuation, stop-words, white-space, others)
# ------------------------------------------------------------------------------------------------------

#Text to lower case
doc <- tm_map(doc, content_transformer(tolower))

# Remove numbers
doc <- tm_map(doc, removeNumbers)

# Remove Punctuations
doc <- tm_map(doc, removePunctuation)

# Remove Stopwords
doc <- tm_map(doc, removeWords, stopwords('english'))

# Remove specific words (example: we should remove the name of the newspaper as it appears on each document)
doc <- tm_map(doc, removeWords, c("reuters","video","image","monday","tuesday",
                                  "wednesday", "thursday","friday","saturday",
                                  "sunday","really","thing","month","year","something"))

# Remove Whitespace
doc <- tm_map(doc, stripWhitespace)

# Stem words ----> I have decided not to do this as it tweaks and cuts the words and they might lose meaning
# doc <- tm_map(doc, stemDocument, language = "english")

# After the above transformations the first document looks like:
inspect(doc[1])

# Remove other punctuation issues
doc <- tm_map(doc, content_transformer(str_remove_all), "[[:punct:]]")

# Check sample document after clean-up
writeLines(as.character(doc[[40]]))

# ------------------------
# Step 16: Lemmatization
# ------------------------
# Another way to reduce the number of inflectional forms of different terms, instead of deconstructing and then 
# trying to rebuild the words, is morphological analysis with the help of a dictionary. 
# This process is called lemmatization, which looks for lemma (the canonical form of a word) instead of stems.
doc <- tm_map(doc, content_transformer(lemmatize_strings))

# Check again
writeLines(as.character(doc[[40]]))

# --------------------------------------------
# Step 17: Create Document Term Matrix (DTM)
# --------------------------------------------
# To analyze the textual data, we use a Document-Term Matrix (DTM) representation: documents as the rows, 
# terms/words as the columns, frequency of the term in the document as the entries. 
# Because the number of unique words in the corpus the dimension can be large.
# Also I enforced lower and upper limit to length of the words included (between 5 and 20 characters)
dtm <- DocumentTermMatrix(doc, control=list(wordLengths=c(5, 20)))
inspect(dtm) # Sparsity=100% and term length=20

# -----------------------------------------------------------------------
# Step 18: remove all terms whose sparsity is greater than the threshold
# -----------------------------------------------------------------------
# Since the sparsity is so high, i.e. a proportion of cells with 0s/ cells with other values is too large,
# let's remove some of these low frequency terms
dtm.clean <- removeSparseTerms(dtm, sparse = 0.85) #was 0.9
inspect(dtm.clean) # Sparsity=77% and term length=14

# ----------------------------------------------------------------------------------
# Step 18A: Return all terms that occur more than 20,000 times in the entire corpus
# ----------------------------------------------------------------------------------
# Result is ordered alphabetically, not by frequency
findFreqTerms(dtm.clean,lowfreq=20000)

# --------------------------------------------------------
# Step 18B: Correlation limit inspection and associations
# --------------------------------------------------------
findAssocs(dtm.clean, terms = c("trump","obama","russia", "state"), corlimit = 0.2)			

# -----------------------------------------------------------------------
# Step 18C: We can draw a simple word cloud by min. frequency of 10,000
# -----------------------------------------------------------------------
set.seed(1234)
freq_dtm_clean = data.frame(sort(colSums(as.matrix(dtm.clean)), decreasing=TRUE))
wordcloud(min.freq=10000,rownames(freq_dtm_clean), freq_dtm_clean[,1], max.words=50, colors=brewer.pal(6, "Dark2"))

# ----------------------------------------------------------------
# Step19: Create Clean and Tidy data (document, term and count)
# ----------------------------------------------------------------
df.tidy <- tidy(dtm.clean) # document, term and count


df.word<- df.tidy %>% # term and frequency
    select(-document) %>%
    group_by(term) %>%
    summarize(freq = sum(count)) %>%
    arrange(desc(freq))

# --------------------------------------------------
# Step 20: Word cloud (based on term and frequency)
# --------------------------------------------------
set.seed(1234) # for reproducibility 
wordcloud2(data=df.word, size=1.5, color='random-dark')

# ----------------------------------------------------
# Step 21: Word cloud for the Fake News (category==0)
# ----------------------------------------------------
set.seed(1234)
df.tidy %>% 
    inner_join(news_combo, by = c('document' = 'ID')) %>% 
    select(-text) %>%
    group_by(term, category) %>%
    summarize(freq = sum(count)) %>%
    filter(category == 0) %>%
    select(-category) %>%
    arrange(desc(freq)) %>%
    wordcloud2(size = 1.5,  color='random-dark')

# ---------------------------------------------------------
# Step 22: Word cloud for the True/Real News (category==1)
# ---------------------------------------------------------
set.seed(1234)
df.tidy %>% 
    inner_join(news_combo, by = c('document' = 'ID')) %>% 
    select(-text) %>%
    group_by(term, category) %>%
    summarize(freq = sum(count)) %>%
    filter(category == 1) %>%
    select(-category) %>%
    arrange(desc(freq)) %>%
    wordcloud2(size = 1.5,  color='random-dark')

# ---------------------------------
# Step 23: Convert dtm to a matrix
# ---------------------------------
dtm.mat <- as.matrix(dtm.clean)
dim(dtm.mat) # numbers of rows and columns respectively: 44267 and 67

# ---------------------------
# Step 24: Setup DataFrame
# ---------------------------
dtm.mat <- cbind(dtm.mat, category = news_combo$category)
dtm.mat[1:10, c(1, 2, 3, ncol(dtm.mat))]

summary(dtm.mat[,'category']) #max between 1 and 2

#as.data.frame(dtm.mat) %>% count(category)
#news %>% count(category) 

# ---------------------------------------
# Step 25: Convert matrix to data frame
# ---------------------------------------
dtm.df <- as.data.frame(dtm.mat)

# -------------------------------------------------------------------------------------
# Step 26: Replace values in category with original values (1 with 0 & 2 with 1)
# -------------------------------------------------------------------------------------
dtm.df$category <- ifelse(dtm.df$category == 2, 1, 0)
dtm.df$category <- as.factor(dtm.df$category)
table(dtm.df$category) # Category 0 (Fake): 22851 and Category 1 (True): 21416

# -----------------------------------------------
# Step 27: Create 80:20 split for train and test
# -----------------------------------------------
set.seed(1234)
index <- sample(nrow(dtm.df), nrow(dtm.df)*0.8, replace = FALSE)
train_news <- dtm.df[index,]
test_news <- dtm.df[-index,]

# ----------------------------------------------------
# Step 28: transfer column names to naming convention
# ----------------------------------------------------
names(train_news) <- make.names(names(train_news))
names(test_news) <- make.names(names(test_news))
table(train_news$category)
table(test_news$category)

# --------------------------------------
# Step 29: Training - Naive Bayes Model
# --------------------------------------
model_nb <- naive_bayes(category ~ ., data = train_news)

summary(model_nb) # model summary

attach(model_nb)  # plot NB model
par(mfrow=c(2,2)) # plot NB model
plot(model_nb)    # plot NB model

model_nb

# ----------------------------------------------
# Step 30: Training - Logistic Regression Model
# ----------------------------------------------
# The algorithm hits the maximum number of allowed iterations before signalling convergence
# Therefore, I have increased the maxit from default of 25 to 100
model_lr <- glm(formula = category ~.,
                data = train_news,
                family = 'binomial',
                control = list(maxit = 100)) 

summary(model_lr) # model summary

set.seed(1234)
attach(model_lr)  # plot LR model
par(mfrow=c(2,2)) # plot LR model
plot(model_lr)    # plot LR model


# ----------------------------------------
# Step 31: Training - Random Forest Model
# ----------------------------------------
k <- round(sqrt(ncol(train_news)-1))
set.seed(1234)
model_rf <- randomForest(formula = category ~ ., 
                         data = train_news,
                         ntree = 100,
                         mtry = k,
                         method = 'class',
                         parallel=TRUE)

model_rf

summary(model_rf)

set.seed(1234)
par(mfrow=c(1,1)) # plot rf model
plot(model_rf, log='y') # plot rf model
legend("right", colnames(model_rf$err.rate),col=1:4,cex=0.8,fill=1:4) # plot rf model

# Evaluate variable importance for RF model
varImpPlot(model_rf, main="Evaluate variable importance - Random Forest Model")
importance(model_rf)
varImp(model_rf)


# -------------------------
# Step 32: Training - SVM
# -------------------------
model_svm = svm(category~., data = train_news)

summary(model_svm)


# -------------------------
# Step 33: Training - NNET
# -------------------------
model_nnet = nnet(category~., data=train_news, size=5, rang = 0.1,decay = 5e-4,maxit=500)

model_nnet

summary(model_nnet)

print(model_nnet)

plotnet(model_nnet, alpha=0.6)


# ----------------------------------------
# Step 34: Predicted values
# ----------------------------------------
train_news$pred_nb <- predict(model_nb, type = 'class')
train_news$pred_lr <- predict(model_lr, type = 'response')
train_news$pred_nnet <- predict(model_nnet)
train_news$pred_svm <- predict(model_svm)
train_news$pred_rf <- predict(model_rf, type = 'response')



# ----------------------------------------
# Step 35: Predicted Values for test set
# ----------------------------------------
test_news$pred_nb <- predict(model_nb, newdata = test_news)
test_news$pred_lr <- predict(model_lr, newdata = test_news, type = 'response')
test_news$pred_nnet <- predict(model_nnet, newdata = test_news)
test_news$pred_svm <- predict(model_svm, newdata = test_news)
test_news$pred_rf <- predict(model_rf, newdata = test_news, type = 'response')


# ----------------------------------------
# Step 36: Plot ROC Curve for train set
# ----------------------------------------
prediction(as.numeric(train_news$pred_nb), as.numeric(train_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(col = 'red', lwd = 2)

prediction(as.numeric(train_news$pred_lr), as.numeric(train_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'blue', lwd = 2)

prediction(as.numeric(train_news$pred_nnet), as.numeric(train_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'black', lwd = 2)

prediction(as.numeric(train_news$pred_svm), as.numeric(train_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'brown', lwd = 2)

prediction(as.numeric(train_news$pred_rf), as.numeric(train_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'green', lwd = 2)

par(mfrow=c(1,1))
legend(0.6, 0.6, legend=c("Naive Bayes", "Logistic Regression", "Neural Networks", "SVM", "Random Forest"),
       col=c("red", "blue", "black", "brown", "green"), lty = 1, cex = 1, box.lty = 1)
mtext("ROC Curve for train set", line=2, font=2, cex=1.2)

# --------------------------------------
# Step 37: Plot ROC Curve for test set
# --------------------------------------
prediction(as.numeric(test_news$pred_nb), as.numeric(test_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(col = 'red', lwd = 2)

prediction(as.numeric(test_news$pred_lr), as.numeric(test_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'blue', lwd = 2)

prediction(as.numeric(test_news$pred_nnet), as.numeric(test_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'black', lwd = 2)

prediction(as.numeric(test_news$pred_svm), as.numeric(test_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'brown', lwd = 2)

prediction(as.numeric(test_news$pred_rf), as.numeric(test_news$category)) %>%
    performance('tpr', 'fpr') %>%
    plot(add = TRUE, col = 'green', lwd = 2)

par(mfrow=c(1,1))
legend(0.6, 0.6, legend=c("Naive Bayes", "Logistic Regression", "Neural Networks", "SVM", "Random Forest"),
       col=c("red", "blue", "black", "brown", "green"), lty = 1, cex = 1, box.lty = 1)
mtext("ROC Curve for test set", line=2, font=2, cex=1.2)

# --------------------------------------------------------------
# Step 38: Set Threshold for Logistic Regression Model and NNET
# --------------------------------------------------------------
roc(test_news$category, test_news$pred_lr) %>% coords()
test_news$pred_lr <- ifelse(test_news$pred_lr > 0.5, 1, 0)
test_news$pred_lr <- as.factor(test_news$pred_lr)

roc(test_news$category, test_news$pred_nnet) %>% coords()
test_news$pred_nnet <- ifelse(test_news$pred_nnet > 0.5, 1, 0)
test_news$pred_nnet <- as.factor(test_news$pred_nnet)

# ---------------------------
# Step 39: Confusion Matrix
# ---------------------------
conf_nb <- caret::confusionMatrix(test_news$category, test_news$pred_nb)
conf_lr <- caret::confusionMatrix(test_news$category, test_news$pred_lr)
conf_nnet <- caret::confusionMatrix(test_news$category, test_news$pred_nnet)
conf_svm <- caret::confusionMatrix(test_news$category, test_news$pred_svm)
conf_rf <- caret::confusionMatrix(test_news$category, test_news$pred_rf)

# ------------------------------------
# Step 40: Confusion Matrix Heatmap
# ------------------------------------
bind_rows(as.data.frame(conf_nb$table), 
          as.data.frame(conf_lr$table), 
          as.data.frame(conf_nnet$table),
          as.data.frame(conf_svm$table),
          as.data.frame(conf_rf$table)) %>% 
    mutate(Model = rep(c("Naive Bayes", "Logistic Regression", "Neural Networks", "SVM", "Random Forest"), each = 4)) %>%
    ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    labs(x = 'Actual', y = 'Predicted') +
    scale_fill_gradient(low = "#c9f6cd", high = "#8be78b") +
    scale_x_discrete(limits = c('1', '0'), labels = c('1' = 'True', '0' = 'Fake')) +
    scale_y_discrete(labels = c('1' = 'True', '0' = 'Fake')) +
    facet_grid(. ~ Model) +
    geom_text(aes(label = Freq), fontface = 'bold') +
    ggtitle("Confusion Matrix Heatmap")+
    theme(panel.background = element_blank(),
          legend.position = 'none',
          plot.title = element_text(hjust = 0.5),
          axis.line = element_line(colour = "black"),
          axis.title = element_text(size = 14, face = 'bold'),
          axis.text = element_text(size = 11, face = 'bold'),
          axis.text.y = element_text(angle = 90, hjust = 0.5),
          strip.background = element_blank(),
          strip.text = element_text(size = 12, face = 'bold'))

# ----------------------------------------
# Step 41: Accuracy and F1 Score table
# ----------------------------------------
acc <- c(nb = conf_nb[['overall']]['Accuracy'], 
         lr = conf_lr[['overall']]['Accuracy'],
         nnet = conf_nnet[['overall']]['Accuracy'],
         svm = conf_svm[['overall']]['Accuracy'],
         rf = conf_rf[['overall']]['Accuracy'])
precision <- c(nb = conf_nb[['byClass']]['Pos Pred Value'], 
               lr = conf_lr[['byClass']]['Pos Pred Value'], 
               nnet = conf_nnet[['byClass']]['Pos Pred Value'],
               svm = conf_svm[['byClass']]['Pos Pred Value'],
               rf = conf_rf[['byClass']]['Pos Pred Value'])
recall <- c(nb = conf_nb[['byClass']]['Sensitivity'], 
            lr = conf_lr[['byClass']]['Sensitivity'],
            nnet = conf_nnet[['byClass']]['Sensitivity'],
            svm = conf_svm[['byClass']]['Sensitivity'],
            rf = conf_rf[['byClass']]['Sensitivity'])

score<-data.frame(Model = c('Naive Bayes', 'Logistic Regression', 'Neural Networks','SVM', 'Random Forest'),
                  Accuracy = acc,
                  F1_Score = (2 * precision * recall) / (precision + recall),
                  row.names = NULL)

score

# Formatted table
score %>%
    kbl(caption = "Table for Accuracy and F1 Score") %>%
    kable_classic(full_width = T, html_font = "Cambria") %>%
    row_spec(5:5, bold = T, color = "white", background = "green") %>%
    footnote(general = "Random Forest has the highest accuracy and F1 score. ",
    )



