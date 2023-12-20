import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Tokenization and POS Tagging
from nltk.tokenize import word_tokenize
from nltk.corpus import brown

# Tokenize sentence and perform POS tagging
sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
print("Tokenization:", tokens)
print("Part-of-Speech Tagging:", pos_tags)

# N-gram Analysis (Bigrams)
text = brown.words(categories='news')[:1000]
bigrams = list(nltk.ngrams(text, 2))
freq_dist = nltk.FreqDist(bigrams)
print("\nN-gram Analysis (Bigrams with Smoothing):")
for bigram in bigrams:
    print(f"{bigram}: {freq_dist[bigram]}")

# Chunking with Regular Expressions and POS Tags
tagged_sentence = nltk.pos_tag(word_tokenize(sentence))
grammar = r"NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(tagged_sentence)
print("\nChunking with Regular Expressions and POS tags:")
print(result)
