
# 7. Text Preprocessing and Vectorization

In NLP tasks, text data usually cannot be fed directly into machine learning or deep learning models. In toxic comment detection, comments often contain noisy characters, abbreviations, or informal writing styles.

Therefore, before training the model, we performed two important steps:

- **Text Preprocessing**
- **Vectorization**

These steps help transform natural language data into numerical representations so the AI model can understand and process the text more effectively.

## 7.1 Preprocessing

**Why is preprocessing necessary?**

Comments on the internet are usually inconsistent and contain a lot of noise. Users may:

- Spam characters or repeat letters continuously
- Insert usernames, links, or IP addresses
- Make spelling or grammar mistakes

These factors make the data harder for machine learning models to process. If the raw data is kept unchanged, the model may learn meaningless patterns, reducing its ability to generalize on unseen data.

Therefore, before vectorization, we built a `clean()` function to normalize comments into a consistent format, making the dataset cleaner and more suitable for training.

**Preprocessing steps used**

***Replacing special patterns with tokens***

Instead of completely removing URLs or usernames, we replaced them with special tokens to preserve part of the sentence context.

```python
text = re.sub(patterns['URL'], ' [URL] ', text)
text = re.sub(patterns['IP_Address'], ' [IP] ', text)
text = re.sub(patterns['Email'], " [EMAIL] ", text)
text = re.sub(patterns['Mention'], ' [USER] ', text)
text = re.sub(patterns['Hashtag'], ' [HASHTAG] ', text)
```

Example:

"Go to https://example.com now!!!"

becomes:

"Go to [URL] now!!!"

***Removing HTML tags and escape sequences***

```python
text = re.sub(patterns['HTML_Tag'], ' ', text)
text = re.sub(patterns['Escape_Sequence'], ' ', text)
```

Some raw comments contain HTML tags or escape characters such as:

- \r
- \n
- \t

These elements do not carry much semantic meaning, so they are removed to clean the text.

***Removing extra spaces and redundant characters***

```python
text = re.sub(r'\s+', ' ', text).strip()
```

After multiple cleaning steps, comments often contain unnecessary consecutive spaces. Therefore, we normalize the spacing to make the text cleaner and more consistent.

***Tokenization and Lemmatization***

After cleaning the text, the data is tokenized:

```python
tokens = tokenizer.tokenize(text)
```

Each sentence is split into separate tokens.

Next, we apply lemmatization:

```python
lemmatized_tokens = [
    lemmatizer.lemmatize(token)
    for token in tokens
]
```

Lemmatization converts words back to their root forms:

"studies" → "study"  
"hating" → "hate"

This helps reduce unnecessary features while preserving the original meaning of the words.

---

## 7.2 Vectorization

***Why is vectorization necessary?***

After preprocessing, the data is still in text format. However, machine learning models cannot directly understand natural language.

Therefore, comments need to be converted into numerical vectors before being fed into machine learning models.

In this project, we used the TF-IDF Vectorization technique.

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/7_1_tfidf.png style="margin: 0 auto; display: block;"><br/>
  <em>Figure 7.1 TF-IDF</em>
</p>

**How does TF-IDF work?**

The main idea behind TF-IDF is quite intuitive:

A word that appears frequently in a specific comment is considered more important.

However, if that word appears in almost every comment, its importance decreases.

For example:

Words such as:

- "the"
- "is"
- "you"

appear very frequently and do not help distinguish toxic comments from non-toxic ones.

On the other hand, words like:

- "idiot"
- "trash"
- "hate"

usually contain stronger toxic signals, so they receive higher weights.

TF-IDF helps the model focus on discriminative words instead of simply counting word frequency.

In this project, we used `TfidfVectorizer` as follows:

```python
vec = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    strip_accents='unicode',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)
```

Some important parameters:

- `ngram_range=(1,2)`  
  Uses both unigram and bigram features so the model can learn phrases like `"stupid idiot"` instead of only individual words.

- `min_df=3`  
  Removes words that appear too infrequently to reduce noise.

- `max_df=0.9`  
  Removes words that appear too frequently across the dataset.

- `sublinear_tf=True`  
  Reduces the impact of words repeated excessively within the same comment.

### Converting text into vectors

After fitting TF-IDF on the training set:

```python
trn_term_doc = vec.fit_transform(train['clean_comment'])
test_term_doc = vec.transform(test['clean_comment'])
```

Each comment is represented as a numerical vector containing many different features.

These vectors are then used as the input for the machine learning models in the next step.
