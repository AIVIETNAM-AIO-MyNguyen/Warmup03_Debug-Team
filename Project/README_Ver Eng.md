# 3. Introduction to the dataset

Online learning environments are only truly effective when learners feel safe and respected. However, manually moderating thousands of discussions daily is a huge challenge for educational platforms. To address this problem, we chose the Jigsaw Toxic Comment Classification dataset as the training platform for our group's toxic comment detection application.

## 3.1. Overview of the dataset

The Jigsaw Toxic Comment Classification Challenge dataset is provided on the Kaggle platform by Jigsaw, a subsidiary of Alphabet. Essentially, this dataset comprises approximately 159,000 real comments collected from Wikipedia discussion pages. Importantly, these comments have all been labeled by users as either toxic or non-toxic. Jigsaw's core objective in releasing this dataset is to encourage the technology community to build natural language processing models capable of accurately identifying and classifying negative connotations in online text.

## 3.2. Data Column Structure

The dataset is organized in a table format, consisting of 8 columns. The first column is `id`, which serves as a unique identifier for each data sample. The most important column is `comment_text`, containing the entire raw text content of the comment to be included in the analysis model. The remaining columns include `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. These are binary label columns that take the values ​​`0` or `1`, representing the absence or presence of each type of malicious behavior, respectively.

In there:

- **Toxic:** Represents rude, disrespectful, or inflammatory language at a basic level.
- **Severe Toxic:** This is an escalating level of the previous label, encompassing extreme hate speech and deliberate malicious attacks.
- **Obscene:** Focuses on the use of swear words, vulgar language, or terms that are inappropriate according to cultural norms.
- **Threat:** This category includes statements containing violent or threatening behavior that directly harm the physical safety of others.
- **Insult (Insult):** Targets acts of insult, defamation, or undermining the reputation of a specific individual in the discussion.
- **Identity Hate:** This refers to attacks and discrimination based on core characteristics such as race, religion, gender, or sexual orientation.

## 3.3. Reasons for choosing the dataset

We decided to trust this dataset because it provides very high coverage with six distinct toxic nuances. In an academic setting, students not only need to avoid vulgar swear words, but also need protection from personal attacks or identity hatred during debates.

Furthermore, because the data is sourced from Wikipedia, the sentence structure closely resembles that of an educational environment, where users typically write longer paragraphs to share knowledge rather than using short, colloquial sentences like those found on social media. Using a globally standardized dataset like Jigsaw will ensure the model achieves the highest accuracy and robustness when implemented in practice.

# 4. Exploratory Data Analytics (EDA)

After understanding the theoretical structure, the next essential step is to visualize the data to uncover hidden patterns. This analysis helps us shape the data preprocessing strategy and select the most suitable model architecture for the application.

## 4.1. Ratio of harmful and healthy comments

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_1_toxic_vs_nontoxic.png style="margin: 0 auto; display: block;"><br/>
Figure 5.1. Proportion of harmful and wholesome comments
</p>

The first graph reflects the overall balance of the dataset through the ratio between toxic and non-toxic comments. The visualization shows an extremely large disparity, with the group of non-toxic comments accounting for approximately 90% of the total data, while the group containing toxic elements only accounts for about 10%.

This severe imbalance is a real characteristic of social networks but poses a significant challenge for AI. If this ratio is maintained during training, the model will tend to assume all comments are healthy, theoretically achieving high accuracy. Therefore, we are forced to apply data balancing techniques such as resampling or adjusting the loss weight during training.

## 4.2. Distribution of the Six Negative Labels

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_2_label_distribution.png style="margin: 0 auto; display: block;"><br/>
<em>Figure 5.2. Frequency of occurrence of 6 harmful labels</em>
</p>

Delving deeper into the 10% of negative comments, the bar chart showing the distribution of six labels reveals the frequency of occurrence for each type of hate behavior. The label `toxic` overwhelmingly outnumbers the other five, followed by `insult` and `obscene` with fairly similar numbers. Conversely, the three labels `severe_toxic`, `identity_hate`, and `threat` have extremely low frequencies, forming minority groups within the dataset.

This uneven distribution indicates that the majority of online violations stop at the level of rudeness or mutual insults. For online learning applications, the frequent appearance of the `insult` label warns us that we need to focus intensely on preventing belittling and personal attacks among learners in order to protect a healthy debate environment.

## 4.3. Characteristics of Comment Length

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_3_comment_length_distribution.png style="margin: 0 auto; display: block;"><br/>
<em>Figure 5.3. Distribution of comment length (Minimum 400 words)</em>
</p>

The histogram showing character length and word count in each comment provides important technical insights. Most comments are densely concentrated in the short segment, ranging from a few dozen to under two hundred words. However, the histogram also shows a "long tail" to the right, representing posts with unusually long lengths.

This characteristic directly affects the configuration of the `max_length` parameter when tokenizing text. If the limit is too short, the model will omit important context from longer articles. If the limit is too long, the system will waste computational resources processing meaningless spaces (padding tokens) in short comments, slowing down the response speed of the real-time moderation application.

## 4.4. Correlation between labels

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_4_labels_correlation_heatmap.png style="margin: 0 auto; display: block;"><br/>
Figure 5.4. Correlation matrix diagram between 6 toxic labels.
</p>

The Heatmap plot shows the Pearson correlation coefficients between the six toxic labels, providing insight into verbal behavior. The strongest correlations appear between two pairs of labels: `toxic` with `insult`, and `obscene` with `insult`. Conversely, the threat label shows almost no significant correlation with the other labels, standing completely independent in the matrix.

This organic correlation demonstrates that when someone uses obscene language, there is a very high probability that they are intending to insult someone. Technically, the high correlation between labels reinforces the decision to use a multi-label classification model, allowing a comment to trigger multiple labels simultaneously instead of forcing the model to select a single label.

## 4.5. Noise Information

The Jigsaw Toxic Comment Classification Challenge dataset compiles comments from Wikipedia, so it's inevitable that it may contain some sensitive, biased, or redundant information. Specifically, some information like IP addresses or usernames may reveal someone's true identity; internal Wikipedia links like `Wikipedia:...`, `Help:...`, `File:...` have no meaning in contexts other than Wikipedia; and URLs or HTML tags may not be emotionally or malicious, but simply add to the model's junk vocabulary.

Identifying and removing these can help the model avoid overfitting and increase the efficiency of the training process. The following are some examples of noise present in the dataset:

- **Escape Sequence:** These characters are for formatting purposes only, such as line breaks and tabs.
- **Wiki link:** These are internal links within Wikipedia, meaning only on Wikipedia.
- **URL:** HTTP links do not carry emotional or malicious value.
- **Hashtag:** A metadata tag used to group or categorize the topics being mentioned.
- **Email:** the user's email address, disclosing that user's information.
- **IP Address:** The IP address, similar to email, is also personal information that needs to be kept secure.
- **Mention:** Comments in the format `@username` are used to reply to or refer to a specific individual.
- **HTML tag:** Formatting tags that use HTML, have no emotional value.

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_5_noise_information_distribution.png style="margin: 0 auto; display: block;"><br/>
Figure 5.5. Frequency distribution chart of noise information
</p>

# 5. Overall Pipeline of the ToxiGuard AI System

Before diving into the details of data processing and model building, we first look at the entire system as a complete pipeline. This helps visualize how a comment moves through different stages before producing the final prediction.

In general, the Toxic Comment Detection pipeline works as follows:

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/6_pipeline.png style="margin: 0 auto; display: block;"><br/>
  <em>Figure 5. Toxic Comment Detection pipeline </em>
</p>

## 5.1 Input Data

The first step is collecting comment data from the Jigsaw Toxic Comment Classification Challenge dataset.

Each row in the dataset contains a comment along with labels such as:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

One important characteristic of this problem is that a single comment can belong to multiple labels at the same time. For example, a comment may contain both insulting and hateful content. Therefore, this is not a standard classification problem but a multi-label classification task.

## 5.2 Text Preprocessing

Real-world text data, especially online comments, is usually very messy. Users may write entirely in uppercase, spam special characters, use abbreviations, or write in informal ways. Feeding raw comments directly into the model usually leads to poor prediction performance.

Therefore, before training, we perform preprocessing steps to clean the data, including:

- Converting all text to lowercase
- Removing special characters
- Removing non-informative patterns such as URLs, HTML tags, and usernames
- Tokenizing sentences into individual words
- Removing stopwords

This step helps the model focus on meaningful content instead of being distracted by unnecessary noise.

## 5.3 Vectorization

In this project, we use TF-IDF Vectorization to represent each comment as a numerical vector based on the importance of words in the dataset.

Words that appear frequently across almost every comment receive lower weights, while offensive or toxic-related words usually receive higher weights.

## 5.4 Classification Model

After vectorization, the data is fed into machine learning models to learn patterns related to toxic behavior.

In this project:

- Naive Bayes is used as the baseline model
- Logistic Regression is selected as the main model

In the final step, the model outputs probability scores for each toxic label.

Based on predefined thresholds, the system determines which labels are assigned to the comment.

# 6. Text Preprocessing and Vectorization

In NLP tasks, text data usually cannot be fed directly into machine learning or deep learning models. In toxic comment detection, comments often contain noisy characters, abbreviations, or informal writing styles.

Therefore, before training the model, we performed two important steps:

- **Text Preprocessing**
- **Vectorization**

These steps help transform natural language data into numerical representations so the AI model can understand and process the text more effectively.

## 6.1 Preprocessing

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

## 6.2 Vectorization

***Why is vectorization necessary?***

After preprocessing, the data is still in text format. However, machine learning models cannot directly understand natural language.

Therefore, comments need to be converted into numerical vectors before being fed into machine learning models.

In this project, we used the TF-IDF Vectorization technique.

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/7_1_tfidf.png style="margin: 0 auto; display: block;"><br/>
  <em>Figure 6.1 TF-IDF</em>
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

# 7. Model Development
This section analyzes the development process of the baseline model and the main model. For the baseline model, the Naive Bayes algorithm combined with TF-IDF was selected. For the main model, Logistic Regression was used.

## 7.1. Naive Bayes Classification
Naive Bayes is a classification algorithm modeled based on Bayes’ theorem in probability and statistics. The algorithm assumes that features are conditionally independent given the class label.

One of the main advantages of the Naive Bayes model is its fast training and prediction time, while still performing efficiently on large datasets. Naive Bayes is widely applied in many fields such as spam filtering, sentiment analysis, document classification, and many other applications.

## 7.2. Logistic Regression
Logistic Regression is a supervised machine learning algorithm mainly used for binary classification problems where the output is either true/false, yes/no, or spam/non-spam. The algorithm calculates the probability that a data point belongs to a class through the sigmoid function.

After calculating the probability that a sample belongs to a class, the model compares the result with a predefined threshold. If the probability exceeds the threshold, the model classifies the input as class 1; otherwise, it is classified as class 0.

In the toxic comment classification problem, which is inherently a multi-label classification task, Logistic Regression performs effectively due to its ability to predict the probability that a comment belongs to a specific class. However, to allow the model to classify a comment into multiple labels simultaneously, the One-vs-Rest method is required.

## 7.3. One-vs-Rest Classifier
One-vs-Rest (OvR) is a method that converts a multi-class classification problem into multiple binary classification problems. In this approach, each binary classifier distinguishes one specific class from all remaining classes.

For example, consider a three-class classification problem: [red, blue, green]. Using the One-vs-Rest method, this problem is transformed into three independent binary classification tasks:

Binary Classification 1: red vs [blue, green]
Binary Classification 2: blue vs [red, green]
Binary Classification 3: green vs [red, blue]

When classifying a new sample, all models output probability scores. The label corresponding to the model with the highest confidence score is selected as the final prediction.

The disadvantage of this method is that one separate model must be trained for each class. Therefore, if the problem contains a very large number of classes, the training process may become slower.

## 7.4. PR-AUC
PR-AUC is an important evaluation metric in Machine Learning used to assess the performance of binary and multi-class classification models, especially in imbalanced datasets. This metric reflects the model’s ability to classify classes accurately across different probability thresholds. PR-AUC consists of two main components:

**PR
(Precision - Recall):** Precision–Recall describes how a binary classification model behaves when the classification threshold changes. It compares two metrics across multiple thresholds:

- Precision: The proportion of correctly predicted positive samples among all predicted positive samples.
- Recall: The proportion of correctly predicted positive samples among all actual positive samples. It indicates how many real positive samples the model successfully captures.

When the classification threshold changes, Precision and Recall tend to vary inversely (when Precision increases, Recall decreases, and vice versa). The graph illustrating the relationship between these two metrics across all thresholds is called the PR curve.

**AUC (Area Under the Curve):** AUC represents the area under the PR curve, with values ranging from 0 to 1. A higher PR-AUC value (closer to 1) indicates better model performance, meaning the model maintains both high Precision and high Recall simultaneously.

## 7.5. Model Development

### 7.5.1. Baseline Model
As mentioned in the previous sections, Naive Bayes was selected as the baseline model. For the baseline model, only basic preprocessing techniques were applied, including:

- Removing empty data
- Converting uppercase characters to lowercase
- Removing meaningless characters in the text (random punctuation, URLs)

```python
# Get dataset information and check for null values
df_train.info()
df_train.isnull().sum()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 159571 entries, 0 to 159570
Data columns (total 9 columns):
 #   Column         Non-Null Count   Dtype 
---  ------         --------------   ----- 
 0   id             159571 non-null  object
 1   comment_text   159571 non-null  object
 2   toxic          159571 non-null  int64 
 3   severe_toxic   159571 non-null  int64 
 4   obscene        159571 non-null  int64 
 5   threat         159571 non-null  int64 
 6   insult         159571 non-null  int64 
 7   identity_hate  159571 non-null  int64 
 8   clean_text     159571 non-null  object
dtypes: int64(6), object(3)
```

After inspection, the dataset was found to contain no missing values. Therefore, preprocessing continued with lowercase conversion and removal of meaningless characters.

```python
# Clean out https, urls, extra spaces, tabs, newlines
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove extra spaces/newlines/tabs
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df_train['clean_text'] = df_train['comment_text'].apply(clean_text)
```

After cleaning the text, the dataset was split into training and validation sets. In addition, the text data was encoded using TF-IDF with the following parameters:

- stop_words='english'
- ngram_range=(1,2)
- min_df=3
- max_df=0.9

```python
# Train, test split
from sklearn.model_selection import train_test_split

X = df_train['clean_text']
y = df_train[label_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer  = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

# Fit
X_train_word = word_vectorizer.fit_transform(X_train)
X_valid_word = word_vectorizer.transform(X_valid)
```

The Jigsaw dataset is a multi-label dataset, meaning that each comment may belong to multiple labels simultaneously (for example, both toxic and insult). Therefore, a separate model was trained for each label in each iteration. At this stage, the ComplementNB model was used.

```python
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, f1_score

results = []

models = {}

for label in label_cols:

    print(f"\nTraining model for: {label}")

    # Current label
    y_train_label = y_train[label]
    y_valid_label = y_valid[label]

    # Model
    model = ComplementNB(alpha=0.5)

    # Train
    model.fit(X_train_word, y_train_label)

    # Predict
    y_pred = model.predict(X_valid_word)

    # Classification report as dictionary
    report = classification_report(
        y_valid_label,
        y_pred,
        output_dict=True
    )

    # Extract metrics for positive class (class 1)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    support = report['1']['support']
    accuracy = report['accuracy']

    # Save results
    results.append({
        'label': label,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'support': support
    })

    # Save model
    models[label] = model

results_df = pd.DataFrame(results)
results_df = results_df.round(4)

results_df
```

After training, the following results were obtained:

|  | label | precision | recall | f1_score | accuracy | support |
| :---------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| 0 | toxic        | 0.7015 | 0.6643 | 0.6824 | 0.9408 | 3056.0 |
| 1 | severe_toxic | 0.3384 | 0.4143 | 0.3725 | 0.9860 | 321.0 |
| 2 | obscene      | 0.6691 | 0.6507 | 0.6598 | 0.9639 | 1715.0 |
| 3 | threat       | 0.1100 | 0.1486 | 0.1264 | 0.9952 | 74.0 |
| 4 | insult       | 0.5985 | 0.5818 | 0.5900 | 0.9591 | 1614.0 |
| 5 | identity_hate| 0.2550 | 0.2177 | 0.2349 | 0.9869 | 294.0 |

**Evaluation of the results table:**

Toxic label:

- This label contains the largest number of samples, with precision = 0.70, recall = 0.66, and F1-score = 0.68.
- When the model predicts a comment as toxic, approximately 70% of those comments are actually toxic, while the model correctly identifies around 66% of all toxic comments.
- However, the relatively low F1-score indicates that many toxic comments are still not correctly detected.

Obscene label:

- This label contains the second-highest number of samples, with precision = 0.66, recall = 0.65, and F1-score = 0.65.
- Obscene words often appear explicitly and independently, making them easier to distinguish. Therefore, the model achieves relatively good performance for this label.
- However, the F1-score is still not high enough because some offensive words are written in modified forms such as "f*ck", "f u c k", "fuuuuuck", or "f#ck", which complicates the feature encoding process. Therefore, better preprocessing is required in the main model.

Insult label:

- This label has the third-highest number of samples, with precision = 0.59, recall = 0.58, and F1-score = 0.59.
- Since insults are highly context-dependent (for example: "you are clueless"), it is difficult to determine whether a comment is truly insulting. Therefore, additional processing is needed to help the model better understand contextual meaning.

Severe_toxic, identity_hate, and threat labels:

- These labels contain relatively few samples, which directly affects precision, recall, and F1-score.
- Due to the imbalanced nature of the dataset, this issue is unavoidable.

A common observation across all labels is that accuracy remains very high, ranging from 0.94 to 0.98. However, in this problem, accuracy alone is not a reliable metric:

- The dataset is highly imbalanced, with the majority of comments being non-toxic. In such cases, high accuracy (for example: 0.97) may simply indicate that the model predicts most comments as non-toxic, without effectively detecting specific toxic categories.
- To evaluate the model more effectively, other metrics such as precision, recall, F1-score, and PR-AUC should be considered.

### 7.5.2. Main Model
As discussed previously, to achieve better performance, additional preprocessing techniques were applied together with Logistic Regression.

```python
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Initialize TweetTokenizer with repeated punctuation reduction
# Example: !!!!! -> !!!
# Set preserve_case=False to automatically convert text to lowercase during tokenization
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)

# Initialize WordNetLemmatizer for lemmatization
# Example: 'studies' -> 'study'
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Replace entities with special tokens to preserve sentence context
    text = re.sub(patterns['URL'], ' [URL] ', text)
    text = re.sub(patterns['IP_Address'], ' [IP] ', text)
    text = re.sub(patterns['Email'], " [EMAIL] ", text)
    text = re.sub(patterns['Wiki_Link'], " [WIKI_LINK] ", text)
    text = re.sub(patterns['Mention'], ' [USER] ', text)
    text = re.sub(patterns['Hashtag'], ' [HASHTAG] ', text)

    # Remove HTML tags and escape characters completely
    text = re.sub(patterns['HTML_Tag'], ' ', text)
    text = re.sub(patterns['Escape_Sequence'], ' ', text)

    # Remove pure numeric values
    text = re.sub(r'\b\d+ \b', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def custom_tokenizer(text):
    # Receive cleaned text and tokenize it
    tokens = tokenizer.tokenize(text)

    # Lemmatize each token immediately using list comprehension
    # Remove tokens with length = 1 (single punctuation or noisy characters)
    # Keep only '!' and '?' because they may carry sentiment information
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if len(token) > 1 or token in ['!', '?']
    ]

    return lemmatized_tokens
```

To improve the dataset compared to the baseline model, additional NLP libraries such as nltk (Natural Language Toolkit) were used:

- WordNet Package: WordNet is a large lexical database of the English language developed by Princeton University. In this project, the WordNet Lemmatizer was used to convert different forms of a word into its root form (for example: running -> run). This process ensures that the model treats different variants of a word as the same feature instead of independent words.
- TweetTokenizer: TweetTokenizer is a tokenizer specifically designed for social media text. Since the Jigsaw dataset contains Wikipedia comments that often include informal writing styles, using TweetTokenizer helps optimize the text preprocessing stage.

During preprocessing, the following steps were applied:

- Replace entities with special tokens to preserve sentence context.
- Completely remove HTML tags and escape characters.
- Remove purely numeric values.
- Remove extra whitespace.
- Apply lemmatization to reduce words to their root forms.


```python
# Apply cleaning function
df['cleaned_text'] = df['comment_text'].map(clean_text)

# Train / Validation split (80/20) for threshold optimization
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Configure TF-IDF
word_vectorizer = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    min_df=5,            # Keep words appearing at least 5 times
    max_df=0.9,          # Remove words appearing in more than 90% of documents
    ngram_range=(1, 2),  # Keep bigrams to preserve negation meaning (e.g., "not idiot")
    sublinear_tf=True,   # Apply logarithmic scaling to term frequency
)

X_train = word_vectorizer.fit_transform(train_df['comment_text'])
X_val = word_vectorizer.transform(val_df['comment_text'])
```

After preprocessing, the text data was transformed into numerical vectors using TF-IDF vectorization. The TfidfVectorizer was configured with the following parameters:

- ngram_range=(1,2): Extract both single words and two-word phrases from the text. Using bigrams helps preserve semantic meaning more effectively (for example, "you idiot" carries stronger offensive intent than "you" and "idiot" separately).
- min_df=5: Ignore tokens appearing fewer than five times. This helps reduce noise such as misspellings or meaningless rare tokens.
- max_df=0.9: Remove tokens appearing in more than 90% of the documents. These tokens are usually common words such as "the", "is", and "you".

Naive Bayes weighting is a technique used to increase the importance of TF-IDF features. By assigning additional weights to features, toxic-related words are emphasized more strongly, which helps improve classification performance. The method is implemented through the following function:

```python
# Multiply the TF-IDF matrix by the Naive Bayes log-count ratio before Logistic Regression
def get_nb_ratio(x, y):
    # Compute basic Naive Bayes probabilities
    p = x[y == 1].sum(axis=0) + 1
    q = x[y == 0].sum(axis=0) + 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.log(p / q)
```

**Algorithm intuition:**
```python
p = x[y == 1].sum(axis=0) + 1
q = x[y == 0].sum(axis=0) + 1
```
- For a specific class (for example: the toxic class where y == 1), calculate the frequency of words appearing in toxic comments (x[y==1]). This produces word occurrence frequencies such as: "stupid" → high, "idiot" → high, "nice" → low.
- Adding 1 to both numerator and denominator avoids undefined cases such as division by zero or log(0).

```python
np.log(p / q)
```
- The log-count ratio (also called Naive Bayes log-count ratio) measures how frequently a word appears in one group compared to another, then applies the logarithm function. This technique quantifies the importance of a word for a specific class. In other words, the algorithm answers the question: “Is this word more associated with toxic or non-toxic comments?”.
- Example: Suppose the word "idiot" appears 500 times while "nice" appears only 10 times in toxic comments. Then log(50) is positive, indicating that "idiot" is strongly associated with toxic meaning. Conversely, if "nice" appears 1000 times and "idiot" appears only 5 times, then log(0.005) becomes negative, indicating that "nice" is more associated with non-toxic meaning.
- Applying this weighting vector creates stronger features, which helps improve the classification capability of the model.

The model was then trained separately for each label while using PR-AUC to determine the optimal classification threshold.

```python 
# Training loop and threshold optimization for 6 labels
best_thresholds = {}
val_predictions_proba = {}

print("--- Start training for 6 labels ---")
for cls in target_labels:
    y_train_cls = train_df[cls].values
    y_val_cls = val_df[cls].values

    # NAIVE BAYES STEP: Compute NB ratio for the current label
    r = get_nb_ratio(X_train, y_train_cls)

    # Transform feature matrices using Naive Bayes weighting
    X_train_nb = X_train.multiply(r)
    X_val_nb = X_val.multiply(r)

    # LOGISTIC REGRESSION STEP: Train on transformed matrices
    model = LogisticRegression(
        C=4.0,
        dual=False,
        solver='liblinear',
        max_iter=200,
        random_state=42
    )

    model.fit(X_train_nb, y_train_cls)

    # Predict probabilities on validation set
    preds_proba = model.predict_proba(X_val_nb)[:, 1]
    val_predictions_proba[cls] = preds_proba

    # Use PR-AUC to determine the best threshold
    precisions, recalls, thresholds = precision_recall_curve(
        y_val_cls,
        preds_proba
    )

    # Compute PR-AUC score
    pr_auc_score = auc(recalls, precisions)

    # Compute F1-score for each threshold
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )

    # Find threshold with highest F1-score
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Save best threshold
    best_thresholds[cls] = best_thresh

    # Re-evaluate metrics using selected threshold
    final_preds = (preds_proba >= best_thresh).astype(int)

    final_precision = precision_score(
        y_val_cls,
        final_preds,
        zero_division=0
    )

    final_recall = recall_score(
        y_val_cls,
        final_preds,
        zero_division=0
    )

    final_f1 = f1_score(
        y_val_cls,
        final_preds,
        zero_division=0
    )

    # Print evaluation report
    print(f"\n==========================================")
    print(f"LABEL: [{cls.upper()}]")
    print(f"------------------------------------------")
    print(f"  • PR-AUC Score:            {pr_auc_score:.4f}")
    print(f"  • Threshold:               {best_thresh:.4f}")
    print(f"  • Metrics at selected threshold:")
    print(f"    - F1-Score:              {final_f1:.4f}")
    print(f"    - Precision:             {final_precision:.4f}")
    print(f"    - Recall:                {final_recall:.4f}")

print(f"\n==========================================")
print("--- Training completed ---")

print("\n--- Pipeline Completed ---")
print("Optimal threshold lookup table for inference:")
print(best_thresholds)
```

After training, the following results were obtained:
| Label         | PR - AUC | Threshold | F1 - score | Precision | Recall |Best threshold|
|:--------------:|:---------:|:----------:|:-----------:|:----------:|:--------:|:--------:|
|     Toxic     |   0.8828 |    0.2784 |     0.8075 |       0.8313 |    0.7850 |     0.2783|
|  Severe_toxic |   0.4076 |    0.1717 |     0.4807 |       0.4444 |    0.5234 |     0.1716|
|    Obsence    |   0.8878 |    0.2205 |     0.8240 |       0.8405 |    0.8082 |     0.2205|
|     Threat    |   0.4601 |    0.1368 |     0.5200 |       0.5132 |    0.5270 |     0.1367|
|     Insult    |   0.7832 |    0.1450 |     0.7320 |       0.6761 |    0.7980 |     0.1450|
| Identity_hate |   0.4443 |    0.1537 |     0.4753 |       0.5388 |    0.4252 |     0.1536|

With this improved approach, the model achieved significantly better performance compared to the baseline model:
| Label          | F1 score (Baseline model) | F1 score (Main model) |
| :------:       | :------: | :------: |
| toxic          | 0.6824	  | 0.8075   |
| severe_toxic   | 0.3725   | 0.4807   |
| obscene        | 0.6598   | 0.8240   |
| threat         | 0.1264   | 0.5200   |
| insult         | 0.5900   | 0.7320   |
| identity_hate  | 0.2349   | 0.4753   |

**Evaluation of Results:**

Comparing the baseline and main models shows that the F1-score improved across all labels. In particular, labels with fewer samples such as threat, insult, and identity_hate nearly doubled their F1-scores in the main model. This demonstrates that the preprocessing and data cleaning stages contributed significantly to model performance.

Evaluation of Each Label in the Main Model Results:

Toxic label:
- PR-AUC = 0.8828 and F1-score = 0.8075 indicate excellent performance, showing that the model can distinguish toxic comments effectively.
- Precision and Recall are relatively balanced (0.83 and 0.78), meaning the model successfully detects most toxic comments while maintaining a relatively low number of false positives.

Obscene label:
- PR-AUC = 0.8878 and F1-score = 0.8240, with both Precision and Recall above 0.8, indicate strong classification performance.
- Obscene comments often contain explicit and strong language, making them easier for the model to identify. This label also contains the second-largest number of samples in the dataset.

Insult label:
- PR-AUC, F1-score, and Recall are all above 0.7, but Precision remains relatively lower (0.6761), indicating that the model tends to over-predict the insult label.

Low-frequency labels (threat, severe_toxic, identity_hate):
- These labels achieve PR-AUC, F1-score, Precision, and Recall values around 0.4–0.5, which is acceptable given the limited number of samples.
- For the threat label, although threatening language often contains strong keywords, the small number of training samples prevents the model from achieving higher performance.
- For the identity_hate label, Precision is higher than Recall, indicating that the model produces fewer false positives but still misses many actual cases. This issue is likely caused by the rarity of samples and the strong dependence on conversational context, which remains a limitation of the current model.

From the training results, it can be concluded that comprehensive preprocessing techniques contributed significantly to improving performance, including:
- Standardizing text and converting all words to lowercase
- Expanding English contractions
- Removing noisy characters, domains, and URLs

Combined with Naive Bayes weighting to strengthen token importance before training Logistic Regression, the proposed approach achieved substantially better results than the baseline model. This demonstrates that the model can effectively classify comments into multiple toxic categories. However, due to the highly imbalanced nature of the dataset, the model still cannot achieve optimal performance across all labels. In addition, labels that heavily depend on conversational context remain difficult to classify accurately. This limitation should be considered for future improvements.
