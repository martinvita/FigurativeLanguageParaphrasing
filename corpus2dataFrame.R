# setwd("~/FigurativeLanguage")
raw.data <- readLines("metaphor_paraphrase_corpus.txt")
n.lines <- length(raw.data)
prev.blank.line <- T

premID <- c()
premises <- c()
hypoID <- c()
hypotheses <- c()
grades <- c()

hid <- 0
pid <- 0


# doplnit ID premis, ID hypothez, 
# rozdelit na 3 casti
# udelat cross validaci
# natrenovat model


for (i in 1:n.lines) { 
  if (raw.data[i]!="" && prev.blank.line) {
    premise <- raw.data[i]
    prev.blank.line <- F
    pid <- pid + 1
  }
  if (raw.data[i]!="" && substr(raw.data[i],start = (nchar(raw.data[i]) - 1), stop = (nchar(raw.data[i]) - 1))=="#") {
    hypothesis <- raw.data[i]
    hid <- hid + 1
    hypothesis.text <- unlist(strsplit(raw.data[i], split = "#", fixed = T))[1]
    if (substr(raw.data[i],start = 1, stop = 1)==" ") {
      hypothesis.text <- substr(hypothesis.text, start = 2, stop = nchar(hypothesis.text))  
    }
    grade <- as.numeric(unlist(strsplit(raw.data[i], split = "#", fixed = T))[2])
    premID <- c(premID, paste0("premID", pid))
    premises <- c(premises, premise)
    hypoID <- c(hypoID, paste0("hypoID", hid))
    hypotheses <- c(hypotheses, hypothesis.text)
    grades <- c(grades, grade)
    prev.blank.line <- F
  }
  if (raw.data[i]=="") {
    prev.blank.line <- T
  }
    
}

lbs <- ifelse(grades>=2, 1, 0)

complete.df <- data.frame(PremiseID=premID, Premise=premises, HypothesisID=hypoID, Hypothesis=hypotheses, Grade=grades, Label=lbs, stringsAsFactors = F)
complete.df <- complete.df[1:nrow(complete.df)-1,]

set.seed(18)
idx.partition <- 0:743 %/% 62 + 1

shuff <- sample(1:744, size = 744)
finalia <- cbind(complete.df[shuff,], data.frame(Partition=idx.partition))
write.csv(finalia, "dataFinal.csv", row.names = F)
write.csv(shuff, "shuffleFinal.csv", row.names = F)
