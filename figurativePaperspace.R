library(readr)
library(keras)
library(purrr)
library(dplyr)

# read raw corpus (shuffled, with IDs, binarized labels etc.)
raw.corpus <- read.csv("dataFinal.csv", stringsAsFactors = F)

# params for further tuning

FLAGS <- flags(
  flag_integer("vocab_size", 20000),
  flag_integer("max_len_padding", 25),
  
  #word embedding size, i. e. vector dimension
  flag_integer("embedding_size", 300),
  flag_integer("seq_embedding_size", 512)
)

# tokenization -- we will obtain a word index w. r. t. word frequency
tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)
fit_text_tokenizer(tokenizer, x = c(raw.corpus$Premise, raw.corpus$Hypothesis))

# we will use FastText word embeddings, analogously we can use word2vec/GloVe embeddings -- do not forget modify the dimension in FLAGS! # https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec # readr functions used again -- read_table does not work well, use read_table2 instead
fast.text.en <- read_table2("wiki.multi.en.vec", col_names = F, skip = 1, progress = T)

word.index <- tokenizer$word_index

# setting the number of words used in real -- minimum of words provided by tokenizer and no. of words we wanted to use (defined in flagging)
voc.size <- min(length(word.index), FLAGS$vocab_size)

# real word types:
words <- names(word.index)[1:voc.size]

# preparing lookup table: key-value, e. g. word index and word type -- one row added for padding symbol
aux.index.table <- data.frame(Indx=1:(voc.size+1), Wrd=c(words, "PAD_VAL"), stringsAsFactors = F)

# FastText data -- we will set up the name of the first column to "Wrd" -- first column contain words, other columns are doubles
colnames(fast.text.en)[1] <- "Wrd"

# we will enrich the lookup table with corresponding vectors, i. e. we will obtain "index-word-Vectors" table:
aux.join <- left_join(aux.index.table, fast.text.en, by = "Wrd")

# building embedding matrix: i-th word from the tokenizer has its vector on the i-th row of the embedding matrix (3rd and further cols of aux.join)
# more efficient implementation than https://keras.rstudio.com/articles/examples/pretrained_word_embeddings.html -- but we use dplyr
emb.mat <- as.matrix(aux.join[,3:ncol(aux.join)])

# NA values are set to all-zero vectors (words not covered by the FastText pretrained embeddings as well as padded values)
emb.mat[is.na(emb.mat)] <- 0

# tricky solution of different indexing R vs. Python (to match the right row of the emb. matrix)
emb.mat2 <- rbind(rep(0, times = FLAGS$embedding_size), emb.mat)

# texts, i. e. questions are transformed into sequences of numbers -- w. r. t. tokenizer results
prems <- texts_to_sequences(tokenizer, raw.corpus$Premise)
hypos <- texts_to_sequences(tokenizer, raw.corpus$Hypothesis)

# question representations are padded into maxlen, padding value is vocabulary real size + 1
prems1 <- pad_sequences(prems, maxlen = FLAGS$max_len_padding, value = voc.size + 1)
hypos1 <- pad_sequences(hypos, maxlen = FLAGS$max_len_padding, value = voc.size + 1)


#####################
#### KERAS MODEL ####
#####################

looses <- c()
accuracies <- c()

for (i in 1:12) {

input1 <- layer_input(shape = c(FLAGS$max_len_padding))
input2 <- layer_input(shape = c(FLAGS$max_len_padding))

embedding <- layer_embedding(
  input_dim = voc.size + 1,
  output_dim = FLAGS$embedding_size,
  input_length = FLAGS$max_len_padding,
  
  # weights are stored in the embedding matrix
  weights = list(emb.mat2),
  
  # word embeddings are not trained
  trainable = F
)

la <- keras_model_sequential()
la %>% layer_conv_1d(filters = 25, activation = "relu", dilation_rate = 1, kernel_size = 5, padding = "same", input_shape = c(25, 300)) %>%
  layer_max_pooling_1d(pool_size = 2L) %>%
  layer_lstm(units = 20) %>%
  layer_dense(units = 15) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, kernel_initializer = "he_normal", kernel_regularizer = regularizer_l2(0.2))


vector1 <- input1 %>% embedding %>% la
vector2 <- input2 %>% embedding %>% la

lb <- layer_concatenate(list(vector1, vector2)) %>% layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs = list(input1, input2), outputs = lb)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)



# training
model %>%
  fit(
    list(prems1[raw.corpus$Partition!=i,], hypos1[raw.corpus$Partition!=i,]),
    raw.corpus$Label[raw.corpus$Partition!=i],
    batch_size = 64,
    epochs = 130
#    validation_data = list(
#      list(prems1[raw.corpus$Partition==12,], hypos1[raw.corpus$Partition==12,]), raw.corpus$Label[raw.corpus$Partition==12]),
#    callbacks = list(
#      callback_early_stopping(patience = 5),
#      callback_reduce_lr_on_plateau(patience = 3)
#    )
)

#### testing

evalres <- evaluate(model, x = list(prems1[raw.corpus$Partition==i,], hypos1[raw.corpus$Partition==i,]), y=raw.corpus$Label[raw.corpus$Partition==i])
accuracies <- c(accuracies, evalres$acc)
looses <- c(looses, evalres$loos)
}
