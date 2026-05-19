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
