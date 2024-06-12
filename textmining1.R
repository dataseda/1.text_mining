library(dplyr)
library(readxl)
library(tidytext)
library(dplyr)
library(syuzhet)
library(lubridate)
library(ggplot2)
library(scales)
library(readr)
library(class)
library(caret)
library(stringr)
library(tm)
library(textclean)
library(SnowballC)
library(ROSE)
library(reshape2)
library(wordcloud)
library(ggrepel)
library(lexicon)
library(textdata)
library(topicmodels)
library("ldatuning")
library(forcats)
library(tidyverse)
library(quanteda)
library(wordcloud)
library(ggplot2)
library("quanteda")
library(word2vec)
library(textTinyR)
library(data.table) 
library("quanteda.textstats")
library(ggrepel) 
library(plotly) 
library(umap) 
library(cluster)

##########---------------------------------1----reading data

text_df <- read.csv("clean.csv")
rew_df <- text_df$text
rew_df <- tibble(line = 1:1228, text = text_df$text)

# corpus
text.source <- VectorSource(text_df$text)
my_corpus <- VCorpus(text.source)

# Clean the corpus
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), "coffee"))
  return(corpus)
}
clean.rew.corpus <- clean_corpus(my_corpus)

# document-term matrix 
rew.dtm <- DocumentTermMatrix(clean.rew.corpus)

# Convert to matrix format
rew.m <- as.matrix(rew.dtm)

# Check dimensions
dim(rew.m)


#######---------optimal topic calculation
result <- FindTopicsNumber(
  rew.dtm,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result)

#################-----------topic modelling 
set.seed(1234)
lda_model <- LDA(rew.dtm, k = 5, control = list(seed=42))  # Specify the number of topics

# Extract posterior probabilities
posterior_probs <- posterior(lda_model)

# Extract topic-word distributions
topic_word_dist <- topics(lda_model)
head(topic_word_dist)
top_terms <- terms(lda_model, 10) 
print(top_terms)

#write.csv(top_terms, "top_terms2.csv", row.names = FALSE)

################################### ------plot1

text_df %>%
  group_by(placeInfo.name) %>%
  summarize(messages = n_distinct(X)) %>%
  ggplot(aes(messages, placeInfo.name)) +
  geom_col() +
  labs(y = NULL)


lda_model %>%
  tidy() %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()


######plot2
lda_topics <- lda_model %>%
  tidy(matrix = "beta")
lda_topics %>%
  arrange(desc(beta))
top_terms_per_topic <- lda_topics %>%
  group_by(topic) %>%
  top_n(5, beta)

ggplot(top_terms_per_topic, aes(x = reorder(term, beta), y = beta, fill = topic)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ topic, scales = "free") +
  labs(x = "Term", y = "Beta Value", title = "Top 5 Terms per Topic") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


####################################---bigrams
bigrams <- text_df %>%
  mutate(text = tm::removeWords(text, stopwords())) %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)
head(bigrams)


# Filter bigrams containing the word "food"---------FOOD-------
food_bigrams <- bigrams %>%
  filter(str_detect(bigram, "\\bfood\\b"))
head(food_bigrams)

food_bigrams %>%
  count(bigram, sort = TRUE)



###################WORDNET
#deep clean
data <- text_df$text
data <- iconv(data, from = "latin1", to = "UTF-8", sub = "")
data <- tolower(data)
data <- gsub("[[:punct:]]", " ", data)
data <- removeWords(data, stopwords("en"))
data <- stripWhitespace(data)
corpus <- Corpus(VectorSource(data))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation) 
corpus <- tm_map(corpus, removeNumbers)
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "\\|")
corpus<-  tm_map(corpus, removeWords, c("doesn"))
corpus <- tm_map(corpus, removeWords, stopwords("en")) 
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)
# Build a term-document matrix
corpus_dtm <- TermDocumentMatrix(corpus)
dtm_m <- as.matrix(corpus_dtm)

#####################cbow

reviews = text_df$text
cbow_model = word2vec(x = reviews, type = "cbow", dim = 15, iter = 30)
cbow_lookslike <- predict(cbow_model, c("food","chef", "atmosphere", "staff", "menu","sustainability"), type = "nearest", top_n = 20) 
print(cbow_lookslike)
cbow_embedding <- as.matrix(cbow_model) 
cbow_embedding <- predict(cbow_model,  c("sustainability","staff", "food", "atmosphere", "chef"), type = "embedding") 
print(cbow_embedding)

# Create a document-term matrix 
dtm <- DocumentTermMatrix(corpus) 
words <- colnames(as.matrix(dtm)) 
word_list <- strsplit(words, " ") 
word_list <- unlist(word_list) 
word_list <- word_list[word_list != ""] 

# checking embeddings 
cbow_embedding <- as.matrix(cbow_model) 
cbow_embedding <- predict(cbow_model, word_list, type = "embedding") 
cbow_embedding <- na.omit(cbow_embedding)

# Convert cbow_embedding matrix to dataframe
cbow_df <- as.data.frame(cbow_embedding)
print(head(cbow_embedding,10))
library(ggplot2) 
library(ggrepel) 
library(plotly) 
library(umap) 
vizualization <- umap(cbow_embedding, n_neighbors = 20, n_threads = 2) 


df <- data.frame(word = rownames(cbow_embedding), 
                 xpos = gsub(".+//", "", rownames(cbow_embedding)), 
                 x = vizualization$layout[, 1], y = vizualization$layout[, 2], 
                 stringsAsFactors = FALSE) 

plot_ly(df, x = ~x, y = ~y, type = "scatter", mode = 'text', text = ~word) %>% 
  layout(title = "CBOW Embeddings Visualization")


#########skip-gram

skip_gram_model = word2vec(x = data, type = "skip-gram", dim = 15, iter = 100)
skip_embedding <- as.matrix(skip_gram_model) 
skip_embedding <- predict(skip_gram_model, c("food","chef", "atmosphere", "staff", "menu","sustainability"), type = "embedding") 
print(skip_embedding)
skip_lookslike <- predict(skip_gram_model, c("sustainability","staff", "food", "atmosphere", "chef"), type = "nearest", 
                          top_n = 20) 
print(skip_lookslike)

skip_embedding <- as.matrix(skip_gram_model) 
skip_embedding <- predict(skip_gram_model, word_list, type = "embedding") 
skip_embedding <- na.omit(skip_embedding) 
print(skip_embedding)
library(ggplot2) 
library(ggrepel) 
library(plotly) 
library(umap) 
vizualization <- umap(skip_embedding, n_neighbors = 20, n_threads = 2) 

df <- data.frame(word = rownames(skip_embedding), 
                 xpos = gsub(".+//", "", rownames(skip_embedding)), 
                 x = vizualization$layout[, 1], y = vizualization$layout[, 2], 
                 stringsAsFactors = FALSE) 

plot_ly(df, x = ~x, y = ~y, type = "scatter", mode = 'text', text = ~word) %>% 
  layout(title = "Skip Gram Embeddings Visualization")


#################dendogram

library(word2vec)
library(stats)
library(dendextend)

skip_gram_model <- word2vec(x = data, type = "skip-gram", dim = 15, iter = 100)
skip_lookslike <- predict(skip_gram_model, c("food", "chef", "atmosphere", "staff", "menu", "sustainability"), type = "nearest", top_n = 20)
print(skip_lookslike)

specific_words <- c("food", "chef", "atmosphere", "staff", "menu","sustainability")
specific_embeddings <- predict(skip_gram_model, specific_words, type = "embedding")
specific_embeddings <- as.matrix(specific_embeddings)

# Check for missing values in embeddings
sum(is.na(specific_embeddings))

print(specific_embeddings)

# Compute cosine distance matrix
cosine_distance <- function(x) {
  as.dist(1 - tcrossprod(x) / sqrt(rowSums(x^2) %*% t(rowSums(x^2))))
}

distance_matrix <- cosine_distance(specific_embeddings)

# Perform hierarchical clustering
hc <- hclust(distance_matrix, method = "ward.D2")

# Plot the dendrogram
plot(as.dendrogram(hc), main = "Dendrogram of Specific Words (Skip-Gram)", xlab = "Words", ylab = "Height")


# Load dendextend for enhanced plotting
library(dendextend)
dend <- as.dendrogram(hc)

# Customize dendrogram appearance
dend <- dend %>%
  set("labels_cex", 0.7) %>%
  set("branches_k_color", k = 3) %>%
  set("branches_lwd", 1.2) %>%
  set("labels_colors", k = 3) %>%
  set("labels_cex", 0.7)

plot(dend, main = "Enhanced Dendrogram of Specific Words (Skip-Gram)", xlab = "Words", ylab = "Height")


#################################################################
#semantic coherence
topics <- list(
  topic1 = c("food", "chef", "dish", "course", "menu","atmosphere"),
  topic2 = c("sustainability", "organic", "local", "waste", "environment")
)
# Function to calculate cosine similarity
cosine_similarity <- function(vec1, vec2) {
  return(sum(vec1 * vec2) / (sqrt(sum(vec1^2)) * sqrt(sum(vec2^2))))
}

# Function to calculate semantic coherence for a topic
calculate_coherence <- function(topic_words, embeddings) {
  word_pairs <- combn(topic_words, 2)
  coherence <- 0
  valid_pairs <- 0
  for (i in 1:ncol(word_pairs)) {
    word1 <- word_pairs[1, i]
    word2 <- word_pairs[2, i]
    if (word1 %in% rownames(embeddings) && word2 %in% rownames(embeddings)) {
      vec1 <- embeddings[word1, ]
      vec2 <- embeddings[word2, ]
      coherence <- coherence + cosine_similarity(vec1, vec2)
      valid_pairs <- valid_pairs + 1
    }
  }
  if (valid_pairs > 0) {
    coherence <- coherence / valid_pairs
  } else {
    coherence <- NA  # No valid pairs found
  }
  return(coherence)
}
# Calculate coherence for each topic using CBOW embeddings
cbow_coherence <- sapply(topics, calculate_coherence, embeddings = cbow_embedding)

# Calculate coherence for each topic using Skip-Gram embeddings
skip_coherence <- sapply(topics, calculate_coherence, embeddings = skip_embedding)

# Print semantic coherence scores
print("Semantic Coherence using CBOW Embeddings:")
print(cbow_coherence)

print("Semantic Coherence using Skip-Gram Embeddings:")
print(skip_coherence)

############frequency based


# Keywords to search for
keywords <- c("food", "chef", "atmosphere", "sustainability", "staff")

# Function to count occurrences of each keyword in text
count_keyword_occurrences <- function(keyword, text_data) {
  occurrences <- sapply(text_data, function(text) grepl(keyword, text))
  total_occurrences <- sum(occurrences)
  return(total_occurrences)
}

# Compute total counts for each keyword
keyword_counts <- sapply(keywords, function(keyword) count_keyword_occurrences(keyword, text_df$text))

# Print total counts for each keyword
print(keyword_counts)


###################################

library(Rtsne)
stop_words <- stopwords("en")
freq_measured <- data_frame(text = text_df$text) %>% 
  mutate(text = tolower(text)) %>% 
  mutate(text = str_remove_all(text, '[[:punct:]]')) %>% 
  mutate(tokens = str_split(text, "\\s+")) %>%
  unnest(tokens) %>%
  count(tokens) %>%
  filter(!tokens %in% stop_words) %>%
  mutate(freq = n / sum(n)) %>%
  arrange(desc(n))


# Perform t-SNE
tsne_result <- Rtsne(cbow_embedding, dims = 2)

# Scale the frequencies to the range 1:5
freq_scaled <- freq_measured$freq * 5 / max(freq_measured$freq)

# Plot the t-SNE result with scaled frequencies
plot(tsne_result$Y[,1], tsne_result$Y[,2], col = "blue", pch = 20, cex = freq_scaled,
     main = "Word2vec-based Distribution of Topics", xlab = "Dimension 1", ylab = "Dimension 2")
legend("topright", legend = "Frequency", col = "blue", pch = 20, cex = 0.5)

# Add labels for top terms
top_terms <- freq_measured$tokens[1:10]  # Assuming you want to label the top 10 terms
text(tsne_result$Y[1:10, 1], tsne_result$Y[1:10, 2], labels = top_terms, cex =0.8, pos = 3, col = "black")



##############################

# Keywords to search for
keywords <- c("food", "chef", "atmosphere", "staff","sustainability")

# Function to count occurrences of each keyword in text
count_keyword_occurrences <- function(keyword, text_data) {
  occurrences <- sapply(text_data, function(text) grepl(keyword, text))
  total_occurrences <- sum(occurrences)
  return(total_occurrences)
}

# Compute total counts for each keyword
keyword_counts <- sapply(keywords, function(keyword) count_keyword_occurrences(keyword, text_df$text))
# Sort keyword counts by frequency in descending order
sorted_counts <- sort(keyword_counts, decreasing = TRUE)
# Print total counts for each keyword
print(keyword_counts)



# Data for "sustainability"
sustainability_data <- data.frame(
  term1 = rep("sustainability", 20),
  term2 = c("palette", "cooking", "strong", "learn", "photos", "highest", "standout", "producers", "ethos", "remains", "plants", "visually", "comes", "game", "exception", "techniques", "magical", "passion", "care", "ingredients"),
  similarity = c(0.9179802, 0.9017024, 0.8974356, 0.8883595, 0.8812479, 0.8750154, 0.8668475, 0.8661334, 0.8645008, 0.8636286, 0.8632116, 0.8629448, 0.8615193, 0.8614991, 0.8606074, 0.8589203, 0.8584211, 0.8561523, 0.8551461, 0.8544548),
  rank = 1:20
)

# Data for "staff"
staff_data <- data.frame(
  term1 = rep("staff", 20),
  term2 = c("friendly", "professional", "welcoming", "knowledgeable", "engaging", "ambience", "service", "expertise", "extremely", "attentive", "knowledgable", "excellent", "helpful", "warm", "relaxed", "welcome", "member", "trouble", "food", "atmosphere"),
  similarity = c(0.9675288, 0.9578097, 0.9509177, 0.9474991, 0.9404121, 0.9388363, 0.9381110, 0.9351636, 0.9340405, 0.9300273, 0.9210503, 0.9181505, 0.9143643, 0.9142978, 0.9061841, 0.9059810, 0.9054598, 0.9051811, 0.9046264, 0.9028564),
  rank = 1:20
)

# Data for "food"
food_data <- data.frame(
  term1 = rep("food", 20),
  term2 = c("well", "service", "exceptional", "fantastic", "really", "excellent", "experience", "ambience", "outstanding", "great", "executed", "whole", "staff", "superb", "incredible", "setting", "absolutely", "amazing", "friendly", "quality"),
  similarity = c(0.9602119, 0.9467153, 0.9421200, 0.9377297, 0.9337388, 0.9328263, 0.9306394, 0.9200850, 0.9185129, 0.9136569, 0.9074842, 0.9068549, 0.9046264, 0.9045694, 0.9010212, 0.8974798, 0.8956107, 0.8945917, 0.8939968, 0.8919978),
  rank = 1:20
)

# Data for "atmosphere"
atmosphere_data <- data.frame(
  term1 = rep("atmosphere", 20),
  term2 = c("relaxed", "attentive", "informal", "welcoming", "friendly", "ambience", "great", "environment", "setting", "unpretentious", "knowledgable", "staff", "romantic", "well", "simple", "sophisticated", "fantastic", "casual", "intimate", "spot"),
  similarity = c(0.9868384, 0.9411481, 0.9352322, 0.9339580, 0.9339576, 0.9192270, 0.9166853, 0.9134914, 0.9124765, 0.9090597, 0.9059466, 0.9028564, 0.9019189, 0.9014106, 0.9012805, 0.9006436, 0.8999870, 0.8983115, 0.8967339, 0.8951453),
  rank = 1:20
)

# Data for "chef"
chef_data <- data.frame(
  term1 = rep("chef", 20),
  term2 = c("passion", "magical", "meet", "meeting", "suppliers", "created", "team", "taking", "understanding", "personal", "creativity", "chefs", "credit", "operation", "talk", "edge", "talented", "explanations", "create", "explaining"),
  similarity = c(0.9408678, 0.9294361, 0.9274471, 0.9265220, 0.9155152, 0.9144374, 0.9089825, 0.9076425, 0.9070480, 0.9059392, 0.8998860, 0.8990039, 0.8974321, 0.8958347, 0.8886634, 0.8873703, 0.8778549, 0.8736464, 0.8735054, 0.8716003),
  rank = 1:20
)


# Define keywords and their similar terms
keywords_and_similar_terms <- list(
  food = c("well", "service", "exceptional", "fantastic", "really", "excellent", "experience", "ambience", "outstanding", "great", "executed", "whole", "staff", "superb", "incredible", "setting", "absolutely", "amazing", "friendly", "quality"),
  chef = c("chef", "passion", "magical", "meet", "meeting", "suppliers", "created", "team", "taking", "understanding", "personal", "creativity", "chefs", "credit", "operation", "talk", "edge", "talented", "explanations", "create", "explaining" ),
  atmosphere = c("atmosphere", "relaxed", "attentive", "informal", "welcoming", "friendly", "ambience", "great", "environment", "setting", "unpretentious", "knowledgable", "staff", "romantic", "well", "simple", "sophisticated", "fantastic", "casual", "intimate", "spot"),
  staff =c("friendly", "professional", "welcoming", "knowledgeable", "engaging", "ambience", "service", "expertise", "extremely", "attentive", "knowledgable", "excellent", "helpful", "warm", "relaxed", "welcome", "member", "trouble", "food", "atmosphere"),
  sustainability = c("palette", "cooking", "strong", "learn", "photos", "highest", "standout", "producers", "ethos", "remains", "plants", "visually", "comes", "game", "exception", "techniques", "magical", "passion", "care", "ingredients"))

# Compute frequencies of terms in text data
keyword_counts2 <- sapply(names(keywords_and_similar_terms), function(keyword) {
  terms <- keywords_and_similar_terms[[keyword]]
  sum(grepl(paste(terms, collapse = "|"), text_df$text))
})

# Print keyword counts
print(keyword_counts)
print(keyword_counts2)


# Sort keyword counts by frequency in descending order
sorted_counts2 <- sort(keyword_counts2, decreasing = TRUE)
soft_pastel_colors <- c("#FFDAB9")
# Plot sorted keyword counts
barplot(sorted_counts2, 
        names.arg = names(sorted_counts2),
        col = soft_pastel_colors,
        main = "Word Cloud Frequency Frekans Analizi", 
        xlab = "Keyword Groups", 
        ylab = "Frequency")

# Sort keyword counts by frequency in descending order
sorted_counts <- sort(keyword_counts, decreasing = TRUE)
soft_pastel_colors <- c("#FFDAB9")
# Plot sorted keyword counts
barplot(sorted_counts, 
        names.arg = names(sorted_counts),
        col = soft_pastel_colors,
        main = "Word Direct Frequency", 
        xlab = "Keyword Groups", 
        ylab = "Frequency")


##lexicon scores

lexicon <- get_sentiments("afinn")
text_tokens <- text_df %>%
  unnest_tokens(word, text)

# Join with the sentiment lexicon
lexicon_scores <- text_tokens %>%
  inner_join(lexicon, by = "word")
unique(lexicon$value)

# Join text_tokens with lexicon to get lexicon scores
lexicon_scores <- text_tokens %>%
  inner_join(lexicon, by = "word")

####food score

food_data2 <- food_data %>%
  inner_join(lexicon_scores, by = c("term2" = "word")) 

weighted_score_food = food_data2$similarity * food_data2$value

sum_scores_food = sum(food_data2$value, na.rm = TRUE)/1610
final_score_food = 10 / (1 + exp(-sum_scores_food)) - 5

print(final_score_food)

summary(food_data2$rating)


# staff score
staff_data2 <- staff_data %>%
  inner_join(lexicon_scores, by = c("term2" = "word")) 

sum_scores_staff = sum(staff_data2$value, na.rm = TRUE)/876
final_score_staff = 10 / (1 + exp(-sum_scores_staff)) - 5

print(final_score_staff)

summary(staff_data2$rating)

####atmosphere score
atmosphere_data2 <- atmosphere_data %>%
  inner_join(lexicon_scores, by = c("term2" = "word")) 

sum_scores_atmosphere = sum(atmosphere_data2$value, na.rm = TRUE)/956
final_score_atmosphere = 10 / (1 + exp(-sum_scores_atmosphere)) - 5

print(final_score_atmosphere)

summary(atmosphere_data2$rating)

#chef score
# Data for "chef"
chef_data <- data.frame(
  term1 = rep("chef", 20),
  term2 = c("passion", "magical", "meet", "meeting", "suppliers", "created", "team", "taking", "understanding", "personal", "creativity", "chefs", "credit", "operation", "talk", "edge", "talented", "explanations", "create", "explaining"),
  similarity = c(0.9408678, 0.9294361, 0.9274471, 0.9265220, 0.9155152, 0.9144374, 0.9089825, 0.9076425, 0.9070480, 0.9059392, 0.8998860, 0.8990039, 0.8974321, 0.8958347, 0.8886634, 0.8873703, 0.8778549, 0.8736464, 0.8735054, 0.8716003),
  rank = 1:20
)

# List of terms related to "chef"
terms <- c("passion", "magical", "meet", "meeting", "suppliers", "created", "team", "taking", "understanding", "personal", "creativity", "credit", "operation", "talk", "edge", "talented", "explanations", "create", "explaining")

# Manually assign sentiment scores to terms
chef_lexicon_manual <- data.frame(
  word = c("passion", "magical", "meet", "meeting", "suppliers", "created", "team", "taking", "understanding", "personal", "creativity", "chefs", "credit", "operation", "talk", "edge", "talented", "explanations", "create", "explaining"),
  value = c("4", "5", "1", "1", "0", "2", "0", "0", "1", "0", "2", "0", "2", "0", "0", "0", "4", "2", "2", "2")
)

chef_data2 <- chef_data %>%
  inner_join(chef_lexicon_manual, by = c("term2" = "word")) 
str(chef_data2$value)
chef_data2$value <- as.numeric(chef_data2$value)
score_chef <- chef_data2$value
sum_scores_chef <- sum(score_chef, na.rm = TRUE) / 20
final_score_chef <- 10 / (1 + exp(-sum_scores_chef)) - 5
print(final_score_chef)

##sust score

sustainability_data2 <- sustainability_data %>%
  inner_join(lexicon_scores, by = c("term2" = "word")) 
sum_scores_sust = sum(sustainability_data2$value, na.rm = TRUE)/46
final_score_sust = 10 / (1 + exp(-sum_scores_sust)) - 5
print(final_score_sust)
summary(sustainability_data2$rating)

##all topics

print(final_score_chef)
print(final_score_food)
print(final_score_sust)
print(final_score_staff)
print(final_score_atmosphere)

# Define categories and final scores
categories <- c("food","chef", "sustainability", "staff", "atmosphere")
final_scores <- c(final_score_food,final_score_chef, final_score_sust, final_score_staff, final_score_atmosphere)
df_plot <- data.frame(Category = categories, Score = final_scores)

# df
df_plot <- df_plot[order(df_plot$Score, decreasing = TRUE), ]

# plot
ggplot(df_plot, aes(x=categories, y=final_scores)) +
  geom_segment( aes(x=categories, xend=categories, y=0, yend=final_scores)) +
  geom_point( size=5, color="blue", fill=alpha("lightblue", 0.3), alpha=0.7, shape=21, stroke=2) 

#########################emotions

d<-get_nrc_sentiment(data)
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (d,10)


syuzhet_vector <- get_sentiment(data, method="syuzhet")
#transpose
td<-data.frame(t(d))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td[2:253]))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new2<-td_new[1:8,]


#Plot two - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(d[, 1:8]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions", xlab="Percentage"
)

#################

# Create dataframes for the frequencies
group1 <- c(food = 767, chef = 163, atmosphere = 118, staff = 458, sustainability = 17)
group2 <- c(food = 1184, chef = 476, atmosphere = 957, staff = 1109, sustainability = 394)

contingency_table <- rbind(group1, group2)

fisher_result <- fisher.test(contingency_table, simulate.p.value = TRUE)
fisher_result

##############################
##accuracy

food_data2 <- food_data2 %>%
  mutate(
    rating_category = case_when(
      rating >= 1 & rating <= 2 ~ "negative",
      rating == 3 ~ "neutral",
      rating >= 4 & rating <= 5 ~ "positive"
    ),
    value_category = case_when(
      value < 0 ~ "negative",
      value == 0 ~ "neutral",
      value > 0 ~ "positive"
    )
  )
food_data2 <- food_data2 %>%
  mutate(correct = ifelse(rating_category == value_category, 1, 0))

TP <- sum(food_data2$rating_category == "positive" & food_data2$value_category == "positive")
TN <- sum(food_data2$rating_category == "negative" & food_data2$value_category == "negative")
FP <- sum(food_data2$rating_category == "positive" & food_data2$value_category != "positive")
FN <- sum(food_data2$rating_category == "negative" & food_data2$value_category != "negative")

# Precision, Recall, F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- sum(food_data2$correct) / nrow(food_data2)

# Print the results
print(paste("Precision: ", precision))
print(paste("Recall: ", recall))
print(paste("Accuracy: ", accuracy))

#atmosphere
atmosphere_data2 <- atmosphere_data2 %>%
  mutate(
    rating_category = case_when(
      rating >= 1 & rating <= 2 ~ "negative",
      rating == 3 ~ "neutral",
      rating >= 4 & rating <= 5 ~ "positive"
    ),
    value_category = case_when(
      value < 0 ~ "negative",
      value == 0 ~ "neutral",
      value > 0 ~ "positive"
    )
  )

# Step 3: Calculate Accuracy
atmosphere_data2 <- atmosphere_data2 %>%
  mutate(correct = ifelse(rating_category == value_category, 1, 0))

TP <- sum(atmosphere_data2$rating_category == "positive" & atmosphere_data2$value_category == "positive")
TN <- sum(atmosphere_data2$rating_category == "negative" & atmosphere_data2$value_category == "negative")
FP <- sum(atmosphere_data2$rating_category == "positive" & atmosphere_data2$value_category != "positive")
FN <- sum(atmosphere_data2$rating_category == "negative" & atmosphere_data2$value_category != "negative")

# Precision, Recall, F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- sum(atmosphere_data2$correct) / nrow(atmosphere_data2)

# Print the results
print(paste("Precision: ", precision))
print(paste("Recall: ", recall))
print(paste("Accuracy: ", accuracy))
correct_counts <- table(atmosphere_data2$correct)

correct_proportion <- prop.table(correct_counts)
print(correct_counts)

###staff

staff_data2 <- staff_data2 %>%
  mutate(
    rating_category = case_when(
      rating >= 1 & rating <= 2 ~ "negative",
      rating == 3 ~ "neutral",
      rating >= 4 & rating <= 5 ~ "positive"
    ),
    value_category = case_when(
      value < 0 ~ "negative",
      value == 0 ~ "neutral",
      value > 0 ~ "positive"
    )
  )

staff_data2 <- staff_data2 %>%
  mutate(correct = ifelse(rating_category == value_category, 1, 0))

TP <- sum(staff_data2$rating_category == "positive" & staff_data2$value_category == "positive")
TN <- sum(staff_data2$rating_category == "negative" & staff_data2$value_category == "negative")
FP <- sum(staff_data2$rating_category == "positive" & staff_data2$value_category != "positive")
FN <- sum(staff_data2$rating_category == "negative" & staff_data2$value_category != "negative")

# Precision, Recall, F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- sum(staff_data2$correct) / nrow(staff_data2)

# Print the results
print(paste("Precision: ", precision))
print(paste("Recall: ", recall))
print(paste("Accuracy: ", accuracy))
correct_counts <- table(staff_data2$correct)

correct_proportion <- prop.table(correct_counts)
print(correct_counts)


###sust
sustainability_data2 <- sustainability_data2 %>%
  mutate(
    rating_category = case_when(
      rating >= 1 & rating <= 2 ~ "negative",
      rating == 3 ~ "neutral",
      rating >= 4 & rating <= 5 ~ "positive"
    ),
    value_category = case_when(
      value < 0 ~ "negative",
      value == 0 ~ "neutral",
      value > 0 ~ "positive"
    )
  )

sustainability_data2 <- sustainability_data2 %>%
  mutate(correct = ifelse(rating_category == value_category, 1, 0))

TP <- sum(sustainability_data2$rating_category == "positive" & sustainability_data2$value_category == "positive")
TN <- sum(sustainability_data2$rating_category == "negative" & sustainability_data2$value_category == "negative")
FP <- sum(sustainability_data2$rating_category == "positive" & sustainability_data2$value_category != "positive")
FN <- sum(sustainability_data2$rating_category == "negative" & sustainability_data2$value_category != "negative")

# Precision, Recall, F1-score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- sum(sustainability_data2$correct) / nrow(sustainability_data2)

# Print the results
print(paste("Precision: ", precision))
print(paste("Recall: ", recall))
print(paste("Accuracy: ", accuracy))
correct_counts <- table(sustainability_data2$correct)

correct_proportion <- prop.table(correct_counts)
print(correct_counts)


# List of words to search for chef manully


word_list <- c("chef", "passion", "magical", "meet", "meeting", "suppliers", "created", "team", 
               "taking", "understanding", "personal", "creativity", "credit", "operation", 
               "edge", "talented", "explanations", "create", "explaining")
filtered_df <- text_df[grep(paste(word_list, collapse = "|"), text_df$text, ignore.case = TRUE), ]

print(filtered_df)

rating_count <- table(filtered_df$rating)

print(rating_count)

mismatched_predictions <- 31

total_predictions <- 459

accuracy <- ((total_predictions - mismatched_predictions) / total_predictions) * 100

# Print accuracy
print(accuracy)


