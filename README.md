#### Problem Statement

##### Project 10
##### Intensity Analysis (Build your own model using NLP and Python) 

The objective of this project is to develop an intelligent system using NLP to predict the intensity in the text reviews. By analyzing various parameters and process data, the system will predict the intensity where its happiness, angriness or sadness. This predictive capability will enable to proactively optimize their processes, and improve overall customer satisfaction.


#### Purpose of the solving this problem

The purpose of developing an intelligent system using NLP to predict the intensity in text reviews can serve multiple practical applications, including:

##### 1. Enhanced Customer Feedback Analysis

Understanding Emotion and Sentiment: Predicting intensity helps companies understand not just whether a review is positive or negative but also the strength of the sentiment. This is useful for prioritizing urgent customer issues (e.g., highly negative reviews) or identifying highly satisfied customers for engagement.

Improving Customer Service: By assessing the intensity of reviews, companies can automatically flag highly intense negative reviews for faster resolution, helping to improve customer retention and overall satisfaction.

##### 2. Improved Product Development

Prioritizing Feedback: Companies can focus on features or products that evoke strong positive or negative reactions. Reviews with higher emotional intensity might signal critical areas to focus on for improvement or enhancement.

Product and Service Customization: Understanding how strongly customers feel about certain aspects of a product can guide product customization to align better with customer preferences.

##### 3. Automated Review Moderation

Identifying Harmful or Fake Reviews: Reviews with extreme or unusual levels of intensity might indicate spam, fake, or abusive reviews. The system can help automatically flag these reviews for further investigation or filtering.

Filtering Content: Intensity prediction can aid in automating moderation by detecting emotionally charged or inappropriate content in public forums or review platforms.

##### 4. Market Sentiment and Trend Analysis

Tracking Sentiment Over Time: Predicting intensity can help businesses understand trends in customer sentiment, monitoring changes in customer satisfaction or dissatisfaction over time.

Competitive Analysis: Companies can analyze the intensity of reviews for their competitors’ products, gaining insight into how strongly customers react to competitor offerings, which can influence strategic decisions.

##### 5. Mental Health and Well-being Insights

Detecting Emotional Distress: In certain applications, such as mental health support platforms, intensity prediction can help detect emotional distress in written feedback or reviews, leading to timely intervention.

Empathy and Interaction: Intelligent systems can be designed to respond with an appropriate tone based on the predicted emotional intensity, enhancing user interaction in support or conversational agents.

In summary, the purpose of solving this problem is to enable deeper insight into the emotional or sentiment-driven aspects of text reviews. This can empower businesses and platforms to enhance customer experience, refine products, manage reputation, and respond more effectively to user feedback.

#### Dataset Information

The dataset for this project can be accessed by clicking the link provided below

https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated Project guide data set/Intensity_data.zip

This data set includes 3 csv files named angriness.csv, happiness.csv and sadness.csv each conatining 2 columns namely 

content : Contains the customer reviews

intensity : Describes the sentiment in the review which is angriness in angriness.csv, sadness in sadness.csv, happiness in happiness.csv

#### Instructions on how to run this project code on your system

Step 1 : Download this project repository as a zip file.
Step 2 : Unzip the folder to your desired location
Step 3 : Go to the Anaconda website and download the installer for your operating system (Windows, macOS, or Linux) & launch it.
Step 4 : Launch Jupyter Notebook Interface from Anaconda Navigator after whic, it opens in your default browser.
Step 5 : Navigate to this project folder.
Step 6 : When inside navigate to intensityanalysis > notebooks > intensityanalysis.pynb
Step 7 : Open the intensityanalysis.pynb
Step 8 : Replace the filepaths to store the data, models and visuals in the directory where you have unzipped the project folder
Step 9 : Save and Run the intensityanalysis.pynb file.
Step 10 : Wait for the file to complete executing the program and then check the output along with the contents in the data, models and visuals directories.


#### Explanations of code and models

##### The comments mentioned throughout the notebook already explain the code and the models. However for the purpose of detailed understanding here is the detailed explanation where I have explained each and every line of code as per the sequence of the notebook.

#### A) Loading the dataset, performing Exploratory Data Analysis and Feature Engineering

##### I) Pandas Cloumn Configuration 

The code pd.set_option('display.max_colwidth', None) is a pandas configuration setting that changes the way DataFrame columns are displayed in the console or notebook.

pd.set_option(): This is a function from pandas that allows you to modify certain display options.
'display.max_colwidth': This option controls the maximum width (in characters) for the display of each column in a DataFrame when printed.

None: Setting it to None means there is no limit on the width. The full content of a column will be displayed, no matter how long the values are.

Purpose:
By default, pandas truncates the content of cells if it exceeds a certain length, showing an ellipsis (...). By setting max_colwidth to None, you can ensure that the entire content of each cell is displayed without truncation.

##### II) Loading the datasets and checking them

df1 = pd.read_csv("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/rawdata/Intensity_data/angriness.csv")
Loads angriness.csv into df1

df1.head() - Shows the first 5 rows of df1

df1.shape - Shows the shape of df1

df1.duplicated().sum() - checks the total duplicates in df1

df2 = pd.read_csv("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/rawdata/Intensity_data/happiness.csv")
Loads happiness.csv into df1

df2.head() - Shows the first 5 rows of df2

df2.shape - Shows the shape of df2

df2.duplicated().sum() - checks the total duplicates in df2

df3 = pd.read_csv("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/rawdata/Intensity_data/sadness.csv")
Loads sadness.csv into df3

df3.head() - Shows the first 5 rows of df3

df3.shape - Shows the shape of df3

df3.duplicated().sum() - checks the total duplicates in df3

df = pd.concat([df1, df2, df3], ignore_index=True) - Appends the datasets (concatenates them row-wise)

##### III) Specify File Paths and Save DataFrame

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "original_appended_data.csv")
Specifies the file path where you want to save the file

df.to_csv(file_path, index=False)
Saves the df dataframe as a CSV file and index=False to helps to avoid saving the DataFrame index

df.head() - Shows the first 5 rows of df

df.shape - Shows the shape of df

df.duplicated().sum() - checks the total duplicates in df

##### IV) Explorartory Data Analysis

df.info()
This provides a concise summary of the df DataFrame, including the number of entries, column names, data types, and non-null counts.

df.isna().sum()/len(df) * 100
This line calculates the percentage of missing (NaN) values in each column of the DataFrame df.

df.isna().sum() counts the number of missing values per column.

Dividing by len(df) (the total number of rows) and multiplying by 100 gives the percentage of missing values for each column.

df.duplicated().sum() - checks the total duplicates in df

visuals_folder = "C:/Users/nikde/Documents/UpGrad/intensityanalysis/visuals"
Defines a folder to save visualizations

##### Plotting the target class distribution:

label_dist = df['intensity'].value_counts().to_dict()
fig = plt.figure(figsize = (10, 5))
ax = plt.bar(label_dist.keys(), label_dist.values(), width=0.25)
plt.xticks([0,1,2])
plt.xlabel("Sentiment")
plt.ylabel("Content Count")
plt.savefig(os.path.join(visuals_folder, "initial_sentiment_distribution.png"))
plt.show()


df['intensity'].value_counts(): Counts the occurrences of each unique value in the 'intensity' column, which is assumed to be the target or sentiment label (e.g., 0, 1, 2 representing different sentiment categories).

.to_dict(): Converts the resulting counts into a dictionary where the keys are the unique intensity labels, and the values are their corresponding counts.

plt.figure(figsize=(10, 5)): Creates a new figure for the plot with a specified width (10) and height (5).

plt.bar(): Generates a bar plot where the x-axis is the intensity labels (keys of the label_dist dictionary), and the y-axis is their respective counts (values of the dictionary).

width=0.25: Sets the width of each bar in the plot.

plt.xticks([0,1,2]): Specifies the tick labels on the x-axis, which correspond to the sentiment categories (0, 1, 2).

plt.xlabel("Sentiment"): Labels the x-axis as "Sentiment".

plt.ylabel("Content Count"): Labels the y-axis as "Content Count".

plt.savefig(): Saves the generated plot as a PNG image to the specified file path (visuals_folder), with the file name initial_sentiment_distribution.png.

plt.show(): Displays the plot on the screen.

##### 

df['length']=df['content'].apply(lambda x: len(x.split(' ')))
adds a new column for the length of the reveiws stored in df['content']

df.head(10) - Shows the first 10 rows of df

print(round(df[df['intensity']=='angriness']['length'].mean()))
print(round(df[df['intensity']=='happiness']['length'].mean()))
print(round(df[df['intensity']=='sadness']['length'].mean()))

prints the mean length for angriness, happiness and sadness reveiews

##### Plotting the distribution based on the length of the reveiews for each intensity

df[df['intensity']=='angriness']['length'].plot.hist(bins=15, alpha=0.3, label="angriness")
df[df['intensity']=='happiness']['length'].plot.hist(bins=15, alpha=1, label="happiness")
df[df['intensity']=='sadness']['length'].plot.hist(bins=15, alpha=0.3, label="sadness")
plt.xlabel("length")
plt.savefig(os.path.join(visuals_folder, "initial_review_length_distribution.png"))
plt.show()

df[df['intensity']=='angriness']: Filters the DataFrame to include only the rows where the 'intensity' column is equal to 'angriness'.

['length']: Selects the 'length' column from the filtered DataFrame, which presumably contains the lengths of text reviews.

.plot.hist(bins=15, alpha=0.3, label="angriness"): Creates a histogram for the review lengths of the 'angriness' category:

bins=15: Specifies the number of bins to divide the range of lengths into, allowing for a clearer view of the distribution.

alpha=0.3: Sets the transparency of the bars to 30%, allowing overlapping bars from different categories to be seen.

label="angriness": Labels this histogram for the legend.

df[df['intensity']=='happiness']['length'].plot.hist(bins=15, alpha=1, label="happiness")
This line performs the same operation as the previous one but for the 'happiness' category. Here, alpha=1 means the bars will be fully opaque.

df[df['intensity']=='sadness']['length'].plot.hist(bins=15, alpha=0.3, label="sadness")
Similar to the previous lines, this one creates a histogram for the 'sadness' category, with a transparency of 30%.

plt.xlabel("length")
This line labels the x-axis as "length", indicating that the x-axis represents the lengths of the text reviews.

plt.savefig(): Saves the histogram plot as a PNG image in the specified directory (visuals_folder) with the filename initial_review_length_distribution.png.

plt.show(): Displays the plot on the screen.

##### Fetching Stopwords

 STOPWORDS = set(stopwords.words('english'))

stopwords.words('english'): This function call comes from the nltk library (Natural Language Toolkit). It retrieves a list of English stopwords, which are words that typically carry little meaning and are usually removed in text preprocessing. Common examples of stopwords include words like "the," "is," "in," "and," etc.

set(...): The set function converts the list of stopwords into a set. Using a set is beneficial because:

Faster Lookup: Sets provide faster membership testing compared to lists. This means checking if a word is a stopword will be quicker.

No Duplicates: A set automatically eliminates any duplicate entries, ensuring that each stopword is only represented once.

Purpose:

The purpose of creating the STOPWORDS set is to prepare for text preprocessing tasks, such as tokenization or text classification, where you may want to exclude these common words to focus on more meaningful words in the text.

##### Removing Stopwords

 df["clean_content"]=df["content"].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

df["content"]: This references the "content" column in the DataFrame df, which presumably contains raw text data (e.g., user reviews, comments, etc.).

.apply(...): The apply function is used to apply a function to each element of the Series (in this case, each row in the "content" column).

lambda x: ...: This defines an anonymous (lambda) function that takes an input x, representing a single piece of text from the "content" column.

x.split(): This method splits the string x into a list of words (tokens) based on whitespace. For example, the sentence "This is a test." would become ["This", "is", "a", "test."].

[word for word in x.split() if word not in STOPWORDS]: This list comprehension iterates over each word in the list created by x.split(). It constructs a new list that includes only the words that are not in the STOPWORDS set. Essentially, it filters out any stopwords.

' '.join(...): After filtering, the join method concatenates the remaining words back into a single string, with a space ' ' between each word.

df["clean_content"] = ...: The resulting cleaned text (with stopwords removed) is assigned to a new column named "clean_content" in the DataFrame df.

Purpose:
The purpose of this code is to preprocess the text data in the "content" column by removing common, less meaningful words (stopwords). The cleaned text is stored in a new column ("clean_content"), which can be used for further analysis or modeling tasks, such as sentiment analysis or text classification.

df.head(10) - Shows the first 10 rows of df again after removing stopwords

##### defining a function for content cleaning

def text_clean(content):
    #lowercase the content
    content=content.lower()
    #remove punctuation
    content = re.sub('[()!?]', ' ', content)
    content = re.sub('\[.*?\]',' ', content)
    #remove non alphanumeric occurences
    content = re.sub("[^a-z0-9]"," ", content)
    #remove the @mention
    content = re.sub("@[A-Za-z0-9_]+","", content)
    #remove the hashtags
    content = re.sub("#[A-Za-z0-9_]+","", content)
    #remove any links
    content = re.sub(r"http\S+", "", content)
    content = re.sub(r"www.\S+", "", content)
    return content
    
Lowercase the content: Converts all characters in the text to lowercase to ensure uniformity. This helps avoid case-sensitive mismatches during further processing (e.g., 'Hello' and 'hello' will be treated as the same word).

Remove specific punctuation: The first re.sub removes parentheses (), exclamation marks !, question marks ?, and replaces them with spaces.The second re.sub removes any text inside square brackets [ ], along with the brackets themselves.

Remove non-alphanumeric characters: This re.sub replaces any character that is not a lowercase letter (a-z) or a digit (0-9) with a space. This effectively removes punctuation and special characters while keeping words and numbers intact.

Remove mentions: This line removes any Twitter-like mentions (e.g., @username) from the text.

Remove hashtags: Similar to mentions, this line removes hashtags (e.g., #hashtag) from the text.

Remove links: The first re.sub removes any URLs that start with http, while the second removes URLs that start with www..
python

Return the cleaned content: Finally, the function returns the cleaned version of the text.

##### Applying the text_clean function and subsequent cleaning

df['clean_content'] = df['clean_content'].apply(text_clean)
applies the function on the clean_content column

df.head(10) - Shows the first 10 rows of df after cleaning

df = df.drop('length', axis = 1) - drops the length column

df.head() - checks df for the change

df['length']=df['clean_content'].apply(lambda x: len(x.split(' '))) 
adds back the length column after the clean_content column for the length of the cleaned reveiws

df.head(5) - checks df for the change

df['clean_content'] - shows the contents of the clean_content column

df["clean_content"]=df["clean_content"].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

removes the stop words again because of new words created after applying the text_clean function

df["clean_content"] - shows the contents of the clean_content column again

df.head() - checks df again after removing stopwords one more time

df['length']=df['clean_content'].apply(lambda x: len(x.split(' '))) 
updates the length column after removing stopwords again for the length of the cleaned reveiws

df.head() - checks df again for the length of the cleaned reviews

df['clean_content'] = df['clean_content'].apply(text_clean) 
applies the text clean method again on the clean_content column 

df['clean_content'] - shows the contents of the clean_content column again

df.head() - inspects the first 5 rows of df

df['length']=df['clean_content'].apply(lambda x: len(x.split(' ')))
updates the length of the reveiws after removing stopwords and applying the text_clean method

df.head() - inspects the first 5 rows of df lastly

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "initial_feature_engineered_data.csv")
Specifies the file path where you want to save the file

df.to_csv(file_path, index=False)  
Saves the df dataframe as a CSV file after feature engineering
Sets the index=False to avoid saving the DataFrame index
    
##### Displaying the top 50 words by frequency

top_words = 50
Setting the number of top words: This variable defines how many of the most common words you want to extract from the dataset.

words = nltk.tokenize.word_tokenize(df['clean_content'].str.cat(sep=' '))

Tokenizing the cleaned content:
df['clean_content'].str.cat(sep=' '): This part concatenates all the entries in the clean_content column of the DataFrame df into a single string, separating each entry with a space.

nltk.tokenize.word_tokenize(...): The word_tokenize function from the Natural Language Toolkit (NLTK) is then used to split this concatenated string into individual words (tokens). This results in a list of words from all the cleaned text in the DataFrame.

filter_words = [word for word in words if word not in STOPWORDS]

Filtering out stopwords:

This line uses a list comprehension to iterate over the list of words.

For each word, it checks whether it is not in the predefined STOPWORDS set (which typically includes common words like "the", "is", "in", etc., that do not carry significant meaning in text analysis).

Only words that are not in the stopwords list are included in the filter_words list.

word_freq = nltk.FreqDist(filter_words)

nltk.FreqDist(...): This function from the NLTK (Natural Language Toolkit) library creates a frequency distribution of the given input. In this case, the input is filter_words, which is a list of words that have already been filtered to exclude common stopwords.

filter_words: This list contains significant words from the cleaned text data after removing stopwords.

Purpose
The purpose of creating a frequency distribution is to count how many times each word appears in the filter_words list. The result, word_freq, will be a dictionary-like object where:

The keys are the unique words from filter_words.
The values are the counts of how many times each word appears.

wordfreq_df = pd.DataFrame(word_freq.most_common(top_words), columns=['Word', 'Frequency'])

word_freq.most_common(top_words):

This method retrieves the top_words most common words from the word_freq frequency distribution. It returns a list of tuples, where each tuple consists of a word and its corresponding frequency count.
                    
pd.DataFrame(...):

This part of the code uses pandas to create a DataFrame from the list of tuples generated by most_common().
The DataFrame is structured in two columns: one for the words ('Word') and one for their frequencies ('Frequency').
columns=['Word', 'Frequency']:

This argument specifies the names of the columns in the resulting DataFrame, making it clear what data each column represents.                    
The purpose of this code is to organize the frequency data into a structured format that can be easily analyzed, manipulated, or visualized. A DataFrame is a convenient way to hold tabular data, and it allows for easy access to specific rows, filtering, sorting, and more.

wordfreq_df - displays the wordfreq_df dataframe

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata","initial_wordfrequency_data.csv")

Specifies the file path where you want to save the file

wordfreq_df.to_csv(file_path, index=False)  
Save the wordfreq_df dataframe as a CSV file
Set index=False to avoid saving the DataFrame index

#####

print(round(df[df['intensity']=='angriness']['length'].mean()))
print(round(df[df['intensity']=='happiness']['length'].mean()))
print(round(df[df['intensity']=='sadness']['length'].mean()))

prints mean length for angriness, happiness and sadness reveiews after cleaning

##### Plotting the distribution based on the length of the reveiews for each intensity after cleaning

The explanation is same as in the previous plot except

plt.savefig(os.path.join(visuals_folder, "reveiw_len_dist_after_feature_engineering.png")) 
Saves the histogram plot as a PNG image in the specified directory (visuals_folder) with the filename reveiw_len_dist_after_feature_engineering.png

#### B) Performing Train Test Split¶

X_train, X_test, y_train, y_test = train_test_split(df["clean_content"],
                                                    df["intensity"],test_size=0.2,
                                                    stratify=df['intensity'])

train_test_split(...):

This function is part of the sklearn.model_selection module and is used to split arrays or matrices into random train and test subsets.
In this case, it splits the DataFrame df into training and testing sets.
Parameters:

df["clean_content"]: This is the feature variable (or independent variable) that will be used for training the model. It contains the cleaned text data.

df["intensity"]: This is the target variable (or dependent variable) that the model will predict. It contains the intensity labels associated with each piece of text.

test_size=0.2: This specifies that 20% of the data should be set aside for testing. The remaining 80% will be used for training the model.

stratify=df['intensity']: This parameter ensures that the split maintains the proportion of each class in the target variable. If, for example, 30% of the data belongs to class A and 70% belongs to class B, the training and testing sets will also reflect this distribution. This is particularly useful when dealing with imbalanced datasets.
Output:

X_train: This variable will contain the training set of the cleaned text content.
X_test: This variable will contain the testing set of the cleaned text content.
y_train: This variable will contain the training set of the intensity labels.
y_test: This variable will contain the testing set of the intensity labels.

Purpose
The purpose of this code is to prepare the data for training a machine learning model. By splitting the data into training and testing sets, you can train the model on one subset of data and evaluate its performance on another, unseen subset. This helps in assessing how well the model generalizes to new data.

print(X_train.shape[0],X_test.shape[0])
checks the shape of train and test data

print(y_train.value_counts())
checks the class balance in the training data

print(y_test.value_counts())
checks the target class balance in the test data

#### C) Applying TF-IDF Vectorization

vectorizertfidf = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))
X_train_tfvec = vectorizertfidf.fit_transform(X_train)
X_test_tfvec = vectorizertfidf.transform(X_test)

TfidfVectorizer(...):

This class is part of the sklearn.feature_extraction.text module and is used to convert a collection of raw documents (text data) into a matrix of TF-IDF features.

use_idf=True: This parameter indicates that the Inverse Document Frequency (IDF) component of the TF-IDF score should be used in the transformation. IDF helps to weigh down the importance of commonly occurring words and gives more weight to rare words.

ngram_range=(1, 2): This parameter specifies the range of n-grams to be extracted. In this case, it will extract both unigrams (1-word sequences) and bigrams (2-word sequences) from the text. This can help capture more context and relationships between words in the documents.

fit_transform(X_train):

This method fits the vectorizer to the training data X_train and transforms the text data into a TF-IDF feature matrix.
The resulting X_train_tfvec will be a sparse matrix where each row corresponds to a document in X_train, and each column corresponds to a feature (word or n-gram) from the text data. The values in the matrix represent the TF-IDF scores of the corresponding words/n-grams in each document.

transform(X_test):

This method transforms the test data X_test into a TF-IDF feature matrix using the vocabulary learned from the training data (i.e., it applies the same transformation to the test set).
The resulting X_test_tfvec will also be a sparse matrix similar to X_train_tfvec, but it will only contain the features that were identified during the fit_transform on the training data. This is important to prevent data leakage, ensuring that the model does not have access to information from the test set during training.

Purpose
The purpose of this code is to prepare the text data for training a machine learning model by converting the textual information into a numerical format that can be fed into the model. TF-IDF is a popular method for representing text data, as it takes into account both the frequency of words in a document and their importance across the entire corpus.

#### D) Training and Evaluation using different Models

##### 1) Creating the score-card and defining a function to update it with important perfromance metric

score_card = pd.DataFrame(columns=['model_name','Accuracy Score','Precision Score','Recall Score','f1 Score'])

def update_score_card(y_test,y_pred,model_name):

    # assign 'score_card' as global variable
    global score_card

    # append the results to the dataframe 'score_card'
    # 'ignore_index = True' do not consider the index labels
    score_card = pd.concat([score_card,pd.DataFrame([{'model_name':model_name,
                                    'Accuracy Score' : accuracy_score(y_test, y_pred),
                                    'Precision Score': precision_score(y_test, y_pred, average="weighted"),
                                    'Recall Score': recall_score(y_test, y_pred, average="weighted"),
                                    'f1 Score': f1_score(y_test, y_pred, average="weighted")}])],
                                    ignore_index = True)
                                    
score_card = pd.DataFrame(columns=['model_name','Accuracy Score','Precision Score','Recall Score','f1 Score'])
This line initializes an empty DataFrame named score_card with predefined columns to store the names and performance metrics of different models.

def update_score_card(y_test,y_pred,model_name):
This function takes three parameters:
y_test: The true labels of the test dataset.
y_pred: The predicted labels from the model.
model_name: A string representing the name of the model being evaluated.

global score_card : This line indicates that the score_card DataFrame used in this function is the same as the one defined outside the function. It allows the function to modify the global variable rather than creating a local copy.


score_card = pd.concat([score_card, pd.DataFrame([{'model_name': model_name,
                                'Accuracy Score': accuracy_score(y_test, y_pred),
                                'Precision Score': precision_score(y_test, y_pred, average="weighted"),
                                'Recall Score': recall_score(y_test, y_pred, average="weighted"),
                                'f1 Score': f1_score(y_test, y_pred, average="weighted")}])],
                                ignore_index=True)
                                
This block performs the following tasks:

It creates a new DataFrame containing the performance metrics calculated using the true and predicted labels:

Accuracy Score: The proportion of correctly predicted instances out of the total instances.
Precision Score: The ratio of true positives to the total predicted positives (weighted by class).
Recall Score: The ratio of true positives to the total actual positives (weighted by class).
F1 Score: The harmonic mean of precision and recall (weighted by class).

It concatenates this new DataFrame with the existing score_card DataFrame and updates the global score_card variable.
ignore_index=True: This ensures that the index is reset and does not retain any previous index labels from the original DataFrame.

Purpose
The purpose of this code is to systematically track the performance of various machine learning models in a structured way. By using the update_score_card function, you can easily evaluate and compare the performance of different models on the same dataset by appending their scores to the scorecard.
                                    
##### 2) Logistic Regression Model

lr = LogisticRegression(class_weight='balanced')
Initializes the Logistic Regression Model

lr.fit(X_train_tfvec,y_train)
Fits the model on the training data

y_pred_lr = lr.predict(X_test_tfvec)
Obtains preditctions on the test data

accuracy_score(y_test,y_pred_lr)
Using accuracy_score() we check the accuracy on the testing dataset

##### confusion matrix

cmlr = confusion_matrix(y_test, y_pred_lr, labels=lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmlr, display_labels=lr.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "LR_Confusion_Matrix.png"))
plt.show()

cmlr = confusion_matrix(y_test, y_pred_lr, labels=lr.classes_)
This computes the confusion matrix for your logistic regression model (lr). It compares the true labels (y_test) with the predicted labels (y_pred_lr).

disp = ConfusionMatrixDisplay(confusion_matrix=cmlr, display_labels=lr.classes_)
This creates a display object for the confusion matrix (disp), which will be plotted later. The matrix will have labels corresponding to the classes in your model.

disp.plot()
This generates a plot of the confusion matrix.

plt.savefig(os.path.join(visuals_folder, "LR_Confusion_Matrix.png"))
Saves the confusion matrix plot as an image (LR_Confusion_Matrix.png) in the specified folder (visuals_folder).

plt.show()
This shows the confusion matrix plot in a window or notebook, depending on your environment.

##### Classification Report and Score Card Update

print(classification_report(y_pred_lr,y_test))
evaluate the performance of logistic regression model on test data using Classification Report

update_score_card(y_test,y_pred_lr,'Initial_LR_model')
calls the update score card method to update the score card with this model's score

##### Performing Hyperparameter tuning

param_gridlr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'newton-cg', 'saga']
}
defines the hyperparameter grid dictionary

grid_searchlr = GridSearchCV(lr, param_gridlr, cv=5, scoring='accuracy')
This creates a GridSearchCV object that will:
Perform cross-validation with cv=5 (5-fold cross-validation) to evaluate the model’s performance.
Optimize accuracy using scoring='accuracy'.
Search over the parameter grid param_gridlr to find the best combination of C and solver for the Logistic Regression model (lr).

grid_searchlr.fit(X_train_tfvec, y_train)
Fits the grid on the training data

best_lr = grid_searchlr.best_estimator_
Obtaining the Best model and stores it in best_lr

##### Training the best model

best_lr - checks the best model
best_lr.fit(X_train_tfvec, y_train) - Fits the best model obtained on the training data
y_pred_blr = best_lr.predict(X_test_tfvec) - Obtains preditctions on the test data

accuracy_score(y_test,y_pred_blr)
Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmblr = confusion_matrix(y_test, y_pred_blr, labels=best_lr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmblr, display_labels=best_lr.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "Tuned_LR_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the tuned LR Model. The code does the same thing as the previous confusion matrix code except that it 
Saves the confusion matrix plot as an image (Tuned_LR_Confusion_Matrix.png) in the specified folder (visuals_folder).

##### Classification Report and Score Card Update

print(classification_report(y_pred_blr,y_test))
evaluates the performance of the tuned logistic regression model on test data using Classification Report

update_score_card(y_test,y_pred_blr,'BestTuned_LR_model')
calls the update score card method to update the score card with this model's score

##### 3) Random Forest Classifier


rf = RandomForestClassifier(n_estimators=200, random_state=0)
Initializes the Random Forest Classifier Model

rf.fit(X_train_tfvec,y_train)
Fits it on the training data

y_pred_rf = rf.predict(X_test_tfvec)
Obtains predictions on the test data

accuracy_score(y_test,y_pred_rf) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmrf = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmrf, display_labels=rf.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "RFC_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the RFC Model. The code does the same thing as the previous confusion matrix code except that it 
Saves the confusion matrix plot as an image (RFC_Confusion_Matrix.png) in the specified folder (visuals_folder).

##### Classification Report and Score Card Update

print(classification_report(y_pred_rf,y_test))
evaluates the performance of the random forest classifier model on test data using classification report

update_score_card(y_test,y_pred_rf,'Initial_RFC_model')
calling the update score card method to update the score card with this model's score

##### Performing Hyperparamter tuning


param_grid_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 50, 100, None]
}
defines the hyperparameter grid dictionary

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')

This creates a GridSearchCV object that will:
Perform 5-fold cross-validation (cv=5) to evaluate the Random Forest model's performance.
Optimize based on accuracy using scoring='accuracy'.
Search through the hyperparameter grid (param_grid_rf) to find the best combination of parameters for the Random Forest model (rf).


grid_search_rf.fit(X_train_tfvec, y_train)
Fits the grid on the training data

best_rf = grid_search_rf.best_estimator_
print(f"Best parameters: {grid_search_rf.best_params_}")
print(f"Best cross-validation accuracy: {grid_search_rf.best_score_}")

This snippet gets and prints the best parameters and model

best_rf - checks the best model

best_rf.fit(X_train_tfvec,y_train) - Fits the best model obtained on the training data

y_pred_brf = rf.predict(X_test_tfvec) - Obtains preditctions on the test data

accuracy_score(y_test,y_pred_brf) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmbrf = confusion_matrix(y_test, y_pred_brf, labels=best_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmbrf, display_labels=best_rf.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "Tuned_RFC_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the Tuned RFC Model. The code does the same thing as the previous confusion matrix code except that it 
Saves the confusion matrix plot as an image (Tuned_RFC_Confusion_Matrix.png) in the specified folder (visuals_folder).

#####

print(classification_report(y_pred_brf,y_test)) - evaluates the performance of the Random Forest Classifier model on test data using Classification Report

update_score_card(y_test,y_pred_brf,'BestTuned_RFC_model') - calls the update score card method to update the score card with this model's score

##### 4) Support Vector Machine Classifier


SVC = LinearSVC()
Initializes the Support Vector Machine Classifier

SVC.fit(X_train_tfvec,y_train)
Fits the model on the training data

y_pred_svc = SVC.predict(X_test_tfvec)
Obtains the predictions on the test data

accuracy_score(y_test,y_pred_svc) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmsvc = confusion_matrix(y_test, y_pred_svc, labels=SVC.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmsvc, display_labels=SVC.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "SVC_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the SVC Model. The code does the same thing as the previous confusion matrix code except that it 
Saves the confusion matrix plot as an image (SVC_Confusion_Matrix.png) in the specified folder (visuals_folder).

##### Classification Report and Score Card Update

print(classification_report(y_pred_svc,y_test))
evaluates the performance of the Support Vector Classifier on test data using Classification Report

update_score_card(y_test,y_pred_svc,'Initial_SVC_model')
calls the update score card method to update the score card with this model's score

##### Performing Hyperparameter tuning


param_grid_svc = {
    'C': [0.01, 0.1, 1, 10, 100]
}
defines the hyperparameter grid dictionary

grid_search_svc = GridSearchCV(SVC, param_grid_svc, cv=5, scoring='accuracy')

This creates a GridSearchCV object that will:
Perform 5-fold cross-validation (cv=5) to evaluate the Support Vector Classifier (SVC) model's performance.
Optimize based on accuracy using scoring='accuracy'.
Search through the hyperparameter grid (param_grid_svc) to find the best combination of parameters for the SVC model.

grid_search_svc.fit(X_train_tfvec, y_train)
Fits the grid on the training data

best_svc = grid_search_svc.best_estimator_
print(f"Best parameters: {grid_search_svc.best_params_}")
print(f"Best cross-validation accuracy: {grid_search_svc.best_score_}")

This snippet Gets and prints the best model and parameters

best_svc - checks the best model
best_svc.fit(X_train_tfvec, y_train) - Fits the best model obtained on the training data

y_pred_bsvc = best_svc.predict(X_test_tfvec) - Obtains the predictions on the test data

accuracy_score(y_test,y_pred_bsvc) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmbsvc = confusion_matrix(y_test, y_pred_bsvc, labels=best_svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmbsvc, display_labels=best_svc.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder, "Tuned_SVC_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the Tuned SVC Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (Tuned_SVC_Confusion_Matrix.png) in the specified folder (visuals_folder).

##### Classification Report and Score Card Update

print(classification_report(y_pred_bsvc,y_test))
evaluate the performance of the tuned Support Vector Classifier model on test data using Classification Report

update_score_card(y_test,y_pred_bsvc,'BestTuned_SVC_model')
calls the update score card method to update the score card with this model's score

#### E) Data Augmentation


df = pd.concat([df1, df2, df3], ignore_index=True)
Appends the original datasets once-again (concatenate them row-wise) for data augmentation

import random
from nltk.corpus import wordnet  - Imports addidtional libraries

nltk.download('wordnet') - Downloads the wordnet lexical database

##### Defining various Functions for augmentation

##### 1) The function get_synonyms(word) retrieves synonyms for a given word using the WordNet lexical database or 
#####  returns an empty list if no synonyms are found.

def get_synonyms(word):
    synonyms = wordnet.synsets(word) 
    if not synonyms:                 
        return []
    return list(set([lemma.name() for synonym in synonyms for lemma in synonym.lemmas()])) 
    
The function get_synonyms(word) retrieves synonyms for a given word using the WordNet lexical database. Here’s a brief breakdown of its functionality:

wordnet.synsets(word): This line fetches all the synsets (sets of synonyms) associated with the input word.

if not synonyms:: It checks if there are no synsets found. If none are found, it returns an empty list.

return list(set([...])): This line constructs a list of unique synonyms:

It iterates over each synonym in the synonyms list.
For each synonym, it retrieves the associated lemmas (word forms).
Using set() ensures that the synonyms are unique, and the result is converted back to a list.

In Summary:
The function returns a list of unique synonyms for a specified word using WordNet, or an empty list if no synonyms are found.

##### 2) The function synonym_replacement(sentence, n) performs synonym replacement in a given sentence. 
##### The function replaces up to n words in a sentence with their randomly selected synonyms, enhancing the variety of the sentence 
##### while retaining its overall meaning.

def synonym_replacement(sentence, n):
    words = sentence.split()          
    new_words = words.copy()          
    
    random_word_list = list(set([word for word in words if get_synonyms(word)]))
    
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:   
                                           
        synonyms = get_synonyms(random_word)
        if synonyms:             
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1    
        if num_replaced >= n:    
            break
             
    return ' '.join(new_words) 

The function synonym_replacement(sentence, n) performs synonym replacement in a given sentence. Here’s a brief breakdown of its functionality:

Input Parameters:

sentence: A string input that represents the sentence in which synonyms will be replaced.
n: An integer specifying the maximum number of words to replace with their synonyms.

Word Splitting:

words = sentence.split(): The sentence is split into a list of individual words.
Copying Words:

new_words = words.copy(): A copy of the original list of words is created to store the modified sentence.

Random Word List:

random_word_list: This list contains unique words from the original sentence that have synonyms available (obtained from the get_synonyms function).

Shuffling:

random.shuffle(random_word_list): The random_word_list is shuffled to randomize the order of words for replacement.

Synonym Replacement:

The function iterates over the shuffled random_word_list, retrieving synonyms for each word:
If synonyms exist, it randomly selects one and replaces the corresponding word in new_words.
The count of replaced words (num_replaced) is incremented.
The loop stops when num_replaced reaches the specified limit n.

Output:

return ' '.join(new_words): The modified list of words is joined back into a single string and returned as the output.

In Summary:

The function replaces up to n words in a sentence with their randomly selected synonyms, enhancing the variety of the sentence while retaining its overall meaning.

##### 3) The function random_insertion(sentence, n) adds random synonyms into a given sentence.
##### The function randomly inserts up to n synonyms from the original sentence into various positions, 
##### enhancing the sentence's richness while retaining its basic structure.

def random_insertion(sentence, n):
    words = sentence.split()    
    new_words = words.copy()    
    for _ in range(n):          
        if len(words) == 0:  
            break

        synonym = get_synonyms(random.choice(words)) 
        if synonym:                             
            insert_pos = random.randint(0, len(new_words))  
            new_words.insert(insert_pos, random.choice(synonym)) 
                                                                 
    return ' '.join(new_words) 


The function random_insertion(sentence, n) adds random synonyms into a given sentence. Here’s a brief breakdown of its functionality:

Input Parameters:

sentence: A string representing the original sentence.
n: An integer specifying the number of synonyms to insert into the sentence.

Word Splitting:

words = sentence.split(): The sentence is split into a list of individual words.

Copying Words:

new_words = words.copy(): A copy of the original list of words is created to store the modified sentence.

Random Insertion Loop:

The loop iterates n times to insert synonyms:
if len(words) == 0:: Checks if there are any words left to choose from. If not, it breaks out of the loop.
synonym = get_synonyms(random.choice(words)): A random word from the original list is selected, and its synonyms are retrieved.

if synonym:: If synonyms are available, a random position is selected for insertion.
new_words.insert(insert_pos, random.choice(synonym)): A randomly chosen synonym is inserted into the new_words list at the selected position.

Output:

return ' '.join(new_words): The modified list of words is joined back into a single string and returned as the output.

In Summary:
The function randomly inserts up to n synonyms from the original sentence into various positions, enhancing the sentence's richness while retaining its basic structure.

###### 4) The function random_swap(sentence, n) randomly swaps pairs of words in a given sentence.

def random_swap(sentence, n):
    words = sentence.split()   
    new_words = words.copy()   
    if len(words) < 2:  
        return sentence

    for _ in range(n):  
        idx1, idx2 = random.sample(range(len(words)), 2)  
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1] 
                                                                            
    return ' '.join(new_words) 
    
##### The function randomly swaps up to n pairs of words in a sentence, altering the sentence's structure while retaining the 
##### original words. If there are fewer than two words, it simply returns the original sentence.

The function random_swap(sentence, n) randomly swaps pairs of words in a given sentence. Here’s a brief breakdown of its functionality:

Input Parameters:

sentence: A string representing the original sentence.
n: An integer specifying the number of pairs of words to swap.
Word Splitting:

words = sentence.split(): The sentence is split into a list of individual words.

Copying Words:

new_words = words.copy(): A copy of the original list of words is created to store the modified sentence.

Check for Minimum Words:

if len(words) < 2:: This condition checks if there are fewer than two words in the sentence. If true, it returns the original sentence since there are not enough words to swap.

Random Swapping Loop:

The loop iterates n times to perform word swaps:
idx1, idx2 = random.sample(range(len(words)), 2): Randomly selects two distinct indices from the list of words.
new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]: The words at the two selected indices are swapped in the new_words list.

Output:

return ' '.join(new_words): The modified list of words is joined back into a single string and returned as the output.

In Summary:
The function randomly swaps up to n pairs of words in a sentence, altering the sentence's structure while retaining the original words. If there are fewer than two words, it simply returns the original sentence.

##### 5) The function random_deletion(sentence, p=0.3) randomly deletes words from a given sentence based on a specified probability. 

def random_deletion(sentence, p=0.3): 
                                      
    words = sentence.split()          
    if len(words) <= 1:  
        return sentence

    new_words = [word for word in words if random.uniform(0, 1) > p] 
    
    if not new_words:
        return words[0]  

    return ' '.join(new_words) 
    
##### The function randomly deletes words from a sentence based on a specified probability. If the sentence has only one word, 
##### it returns the original sentence. If all words are deleted, it returns the first word to prevent an empty output.

The function random_deletion(sentence, p=0.3) randomly deletes words from a given sentence based on a specified probability. Here’s a brief breakdown of its functionality:

Input Parameters:

sentence: A string representing the original sentence.
p: A float representing the probability of deleting each word (default is 0.3, meaning there is a 30% chance for each word to be deleted).

Word Splitting:

words = sentence.split(): The sentence is split into a list of individual words.

Check for Minimum Words:

if len(words) <= 1:: If there is only one word in the sentence, it simply returns the original sentence without any deletion.

Random Deletion:

new_words = [word for word in words if random.uniform(0, 1) > p]: This line creates a new list of words (new_words) that includes words from the original list only if a random number (between 0 and 1) is greater than the probability p. This effectively removes words with a 30% chance of being deleted.

Check for Empty Result:

if not new_words:: If no words remain after deletion, it returns the first word of the original list to avoid returning an empty sentence.

Output:

return ' '.join(new_words): The modified list of words is joined back into a single string and returned as the output.

In Summary:
The function randomly deletes words from a sentence based on a specified probability. If the sentence has only one word, it returns the original sentence. If all words are deleted, it returns the first word to prevent an empty output.


##### 6) The function eda(sentence, num_aug=4) performs various data augmentation techniques on a given sentence to 
#####   generate multiple augmented versions of it

def eda(sentence, num_aug=4): 
    augmented_sentences = []  
    augmented_sentences.append(synonym_replacement(sentence, n=2)) # Replaces two words in the sentence with their synonyms.
    augmented_sentences.append(random_insertion(sentence, n=2))    # Inserts two random synonyms from the sentence into various positions.
    augmented_sentences.append(random_swap(sentence, n=2))         # Swaps two random words in the sentence.
    augmented_sentences.append(random_deletion(sentence, p=0.3))   # Deletes words from the sentence with a probability of 30%.
    return augmented_sentences

##### The function generates a list of augmented versions of a given sentence using various techniques 
##### (synonym replacement, random insertion, random swapping, and random deletion), 
##### enhancing the data for tasks such as natural language processing or machine learning.

The function eda(sentence, num_aug=4) performs various data augmentation techniques on a given sentence to generate multiple augmented versions of it. Here’s a brief breakdown of its functionality:

Input Parameters:

sentence: A string representing the original sentence to be augmented.
num_aug: An integer indicating the number of augmented sentences to create (default is 4).

List Initialization:

augmented_sentences = []: This initializes an empty list to store the augmented sentences.

Augmentation Techniques:

The function applies four different augmentation techniques and appends the results to the augmented_sentences list:

synonym_replacement(sentence, n=2): Replaces two words in the sentence with their synonyms.
random_insertion(sentence, n=2): Inserts two random synonyms from the sentence into various positions.
random_swap(sentence, n=2): Swaps two random words in the sentence.
random_deletion(sentence, p=0.3): Deletes words from the sentence with a probability of 30%.

Output:

return augmented_sentences: The function returns the list of augmented sentences.

In Summary:
The function generates a list of augmented versions of a given sentence using various techniques (synonym replacement, random insertion, random swapping, and random deletion), enhancing the data for tasks such as natural language processing or machine learning.

##### 7) This code augments a dataset by generating new sentences for each entry using different augmentation techniques. 
##### It creates a new DataFrame to hold the original and augmented sentences, 
##### which can be combined with the original data for further analysis or modeling.

##### Augment the dataset
augmented_sentences = [] # Initializes an empty list to store augmented sentences.
augmented_labels = []    # Initializes an empty list to store labels corresponding to the augmented sentences.


for index, row in df.iterrows(): # Iterates through each row in the DataFrame df
    sentence = row['content']    # Retrieves the content (sentence) from the current row.
    label = row['intensity']     # Retrieves the intensity label associated with the current sentence.

    # Generate augmented sentences for each row
    aug_sentences = eda(sentence, num_aug=4) # Calls the eda function to generate four augmented sentences 
                                             # for the current sentence.

    # Append the original and augmented sentences along with labels
    augmented_sentences.append(sentence)  # Adds the original sentence to the augmented_sentences list.
    augmented_labels.append(label)  # Adds the corresponding label for the original sentence to the augmented_labels list.

    # Add augmented data
    augmented_sentences.extend(aug_sentences) # Adds the newly generated augmented sentences to the list.
    augmented_labels.extend([label] * len(aug_sentences)) # Duplicates the original label for each of the augmented sentences 
                                                          # and appends it to the augmented_labels list.

##### Constructs a new DataFrame containing the augmented sentences and their corresponding labels.
augmented_df = pd.DataFrame({
    'content': augmented_sentences,  
    'intensity': augmented_labels
})


##### Combine original and augmented data (optional, if you want to include the original data too)
final_df = pd.concat([df, augmented_df])

##### Prints the final DataFrame containing both original and augmented data.
print(final_df)

In Summary:
This code augments a dataset by generating new sentences for each entry using different augmentation techniques. It creates a new DataFrame to hold the original and augmented sentences, which can be combined with the original data for further analysis or modeling.

##### Checking and saving the datasets generated

augmented_df - checks the augmented_df dataframe

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "augmented_data.csv")
Specifies the file path where you want to save the file

augmented_df.to_csv(file_path, index=False)  
Saves the augmented_df dataframe as a CSV file
Set index=False to avoid saving the DataFrame index

final_df - checks final_df

final_df.shape - checks the shape of final_df

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "original_appended+augmented_data.csv")
Specifies the file path where you want to save the file

final_df.to_csv(file_path, index=False)  
Saves the final_df dataframe as a CSV file
Sets index=False to avoid saving the DataFrame index

#### F) Performing Exploratory Data Analysis and Feature Engineering after Augmentation¶

final_df.info() - This provides a concise summary of the final_df DataFrame, including the number of entries, column names, data types, and non-null counts.

final_df.isna().sum()/len(df) * 100 
This line calculates the percentage of missing (NaN) values in each column of the DataFrame df.

final_df.isna().sum() - counts the number of missing values per column.

Dividing by len(final_df) (the total number of rows) and multiplying by 100 gives the percentage of missing values for each column.

final_df.duplicated().sum() - checks the total duplicates in final_df

visuals_folder_aa = "C:/Users/nikde/Documents/UpGrad/intensityanalysis/visuals/afteraugmentation" 
Defines a folder to save augmented visualizations

##### Plotting the target class distribution:

label_dist = final_df['intensity'].value_counts().to_dict() 
fig = plt.figure(figsize = (10, 5)) 
ax = plt.bar(label_dist.keys(), label_dist.values(), width=0.25) 
plt.xticks([0,1,2]) 
plt.xlabel("Sentiment") 
plt.ylabel("Content Count") 
plt.savefig(os.path.join(visuals_folder_aa, "sentiment_distribution_after_augmentation.png")) 
plt.show()

final_df['intensity'].value_counts(): Counts the occurrences of each unique value in the 'intensity' column, which is assumed to be the target or sentiment label (e.g., 0, 1, 2 representing different sentiment categories).

.to_dict(): Converts the resulting counts into a dictionary where the keys are the unique intensity labels, and the values are their corresponding counts.

plt.figure(figsize=(10, 5)): Creates a new figure for the plot with a specified width (10) and height (5).

plt.bar(): Generates a bar plot where the x-axis is the intensity labels (keys of the label_dist dictionary), and the y-axis is their respective counts (values of the dictionary).

width=0.25: Sets the width of each bar in the plot.

plt.xticks([0,1,2]): Specifies the tick labels on the x-axis, which correspond to the sentiment categories (0, 1, 2).

plt.xlabel("Sentiment"): Labels the x-axis as "Sentiment".

plt.ylabel("Content Count"): Labels the y-axis as "Content Count".

plt.savefig(): Saves the generated plot as a PNG image to the specified file path (visuals_folder_aa), with the file name sentiment_distribution_after_augmentation.png.

plt.show(): Displays the plot on the screen.

#####

final_df['length']=final_df['content'].apply(lambda x: len(x.split(' '))) 
adds a new column for the length of the reveiws stored in df['content']

final_df.head(10) - Shows the first 10 rows of final_df

print(round(final_df[final_df['intensity']=='angriness']['length'].mean()))
print(round(final_df[final_df['intensity']=='happiness']['length'].mean()))
print(round(final_df[final_df['intensity']=='sadness']['length'].mean()))

prints the mean length for angriness, happiness and sadness reviews

##### Fetching stopwords

STOPWORDS = set(stopwords.words('english'))

stopwords.words('english'): This function call comes from the nltk library (Natural Language Toolkit). It retrieves a list of English stopwords, which are words that typically carry little meaning and are usually removed in text preprocessing. Common examples of stopwords include words like "the," "is," "in," "and," etc.

set(...): The set function converts the list of stopwords into a set. Using a set is beneficial because:

Faster Lookup: Sets provide faster membership testing compared to lists. This means checking if a word is a stopword will be quicker.

No Duplicates: A set automatically eliminates any duplicate entries, ensuring that each stopword is only represented once.

Purpose:

The purpose of creating the STOPWORDS set is to prepare for text preprocessing tasks, such as tokenization or text classification, where you may want to exclude these common words to focus on more meaningful words in the text.

##### Removing Stopwords

final_df["clean_content"]=final_df["content"].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

final_df["content"]: This references the "content" column in the DataFrame final_df, which presumably contains raw text data (e.g., user reviews, comments, etc.).

.apply(...): The apply function is used to apply a function to each element of the Series (in this case, each row in the "content" column).

lambda x: ...: This defines an anonymous (lambda) function that takes an input x, representing a single piece of text from the "content" column.

x.split(): This method splits the string x into a list of words (tokens) based on whitespace. For example, the sentence "This is a test." would become ["This", "is", "a", "test."].

[word for word in x.split() if word not in STOPWORDS]: This list comprehension iterates over each word in the list created by x.split(). It constructs a new list that includes only the words that are not in the STOPWORDS set. Essentially, it filters out any stopwords.

' '.join(...): After filtering, the join method concatenates the remaining words back into a single string, with a space ' ' between each word.

final_df["clean_content"] = ...: The resulting cleaned text (with stopwords removed) is assigned to a new column named "clean_content" in the DataFrame final_df.

Purpose: The purpose of this code is to preprocess the text data in the "content" column by removing common, less meaningful words (stopwords). The cleaned text is stored in a new column ("clean_content"), which can be used for further analysis or modeling tasks, such as sentiment analysis or text classification.

final_df.head(10) - Shows the first 10 rows of final_df again

##### Applying the text_clean function and subsequent cleaning

final_df['clean_content'] = final_df['clean_content'].apply(text_clean) applies the function on the clean_content column

final_df.head(10) - Shows the first 10 rows of final_df after cleaning

final_df = final_df.drop('length', axis = 1) - drops the length column

final_df.head() - checks final_df for the change

final_df['length']=final_df['clean_content'].apply(lambda x: len(x.split(' '))) adds back the length column after the clean_content column for the length of the cleaned reveiws

final_df.head(5) - checks final_df for the change

final_df["clean_content"]=final_df["clean_content"].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

removes the stop words again because of new words created after applying the text_clean function

final_df['length']=final_df['clean_content'].apply(lambda x: len(x.split(' '))) updates the length column after removing stopwords again for the length of the cleaned reveiws

final_df.head() - inspects the first 5 rows of final_df lastly

print(round(final_df[final_df['intensity']=='angriness']['length'].mean()))
print(round(final_df[final_df['intensity']=='happiness']['length'].mean()))
print(round(final_df[final_df['intensity']=='sadness']['length'].mean()))

prints the mean length for angriness, happiness and sadness reviews

##### Plotting the distribution based on the length of the reveiews for each intensity after cleaning

final_df[final_df['intensity']=='angriness']['length'].plot.hist(bins=15, alpha=0.3, label="angriness")
final_df[final_df['intensity']=='happiness']['length'].plot.hist(bins=15, alpha=1, label="happiness")
final_df[final_df['intensity']=='sadness']['length'].plot.hist(bins=15, alpha=0.3, label="sadness")
plt.xlabel("length")
plt.savefig(os.path.join(visuals_folder_aa, "reveiw_len_dist_after_augmentation.png"))
plt.show()

The explanation is same as in the non-augmented plot except

plt.savefig(os.path.join(visuals_folder, "reveiw_len_dist_after_augmentation.png")) Saves the histogram plot as a PNG image in the specified directory (visuals_folder_aa) with the filename reveiw_len_dist_after_augmentation.png

final_df.head(10) - inspects the processed dataset final_df

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "final_processed_data.csv")
Specifies the file path where you want to save the file

final_df.to_csv(file_path, index=False)
Saves the final_df dataframe as a CSV file
Set index=False to avoid saving the DataFrame index

##### Displaying the top 50 words by frequency

top_words = 50 Setting the number of top words: This variable defines how many of the most common words you want to extract from the dataset.

words = nltk.tokenize.word_tokenize(final_df['clean_content'].str.cat(sep=' '))

Tokenizing the cleaned content: final_df['clean_content'].str.cat(sep=' '): This part concatenates all the entries in the clean_content column of the DataFrame final_df into a single string, separating each entry with a space.

nltk.tokenize.word_tokenize(...): The word_tokenize function from the Natural Language Toolkit (NLTK) is then used to split this concatenated string into individual words (tokens). This results in a list of words from all the cleaned text in the DataFrame.

filter_words = [word for word in words if word not in STOPWORDS]

Filtering out stopwords:

This line uses a list comprehension to iterate over the list of words.

For each word, it checks whether it is not in the predefined STOPWORDS set (which typically includes common words like "the", "is", "in", etc., that do not carry significant meaning in text analysis).

Only words that are not in the stopwords list are included in the filter_words list.

word_freq = nltk.FreqDist(filter_words)

nltk.FreqDist(...): This function from the NLTK (Natural Language Toolkit) library creates a frequency distribution of the given input. In this case, the input is filter_words, which is a list of words that have already been filtered to exclude common stopwords.

filter_words: This list contains significant words from the cleaned text data after removing stopwords.

Purpose The purpose of creating a frequency distribution is to count how many times each word appears in the filter_words list. The result, word_freq, will be a dictionary-like object where:

The keys are the unique words from filter_words. The values are the counts of how many times each word appears.

final_wordfreq_df = pd.DataFrame(word_freq.most_common(top_words), columns=['Word', 'Frequency'])

word_freq.most_common(top_words):

This method retrieves the top_words most common words from the word_freq frequency distribution. It returns a list of tuples, where each tuple consists of a word and its corresponding frequency count.

pd.DataFrame(...):

This part of the code uses pandas to create a DataFrame from the list of tuples generated by most_common(). The DataFrame is structured in two columns: one for the words ('Word') and one for their frequencies ('Frequency'). 

columns=['Word', 'Frequency']:

This argument specifies the names of the columns in the resulting DataFrame, making it clear what data each column represents.
The purpose of this code is to organize the frequency data into a structured format that can be easily analyzed, manipulated, or visualized. A DataFrame is a convenient way to hold tabular data, and it allows for easy access to specific rows, filtering, sorting, and more.

final_wordfreq_df - displays the final_wordfreq_df dataframe

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata","final_word_frequency_data.csv")
Specifies the file path where you want to save the file

final_wordfreq_df.to_csv(file_path, index=False)
Save the final_wordfreq_df dataframe as a CSV file Set index=False to avoid saving the DataFrame index

#### G) Performing Train Test Split after Augmentation

X_train, X_test, y_train, y_test = train_test_split(final_df["clean_content"],
                                                    final_df["intensity"],test_size=0.2,
                                                    stratify=final_df['intensity'])
                                                    
splits the data into training and test set with balanced split based on target class

print(X_train.shape[0],X_test.shape[0]) checks the shape of train and test data

print(y_train.value_counts()) checks the class balance in the training data

print(y_test.value_counts()) checks the target class balance in the test data

#### H) Applying TF-IDF Vectorization after augmentation

vectorizertfidf = TfidfVectorizer(use_idf=True, ngram_range=(1, 2)) X_train_tfvec = vectorizertfidf.fit_transform(X_train) X_test_tfvec = vectorizertfidf.transform(X_test)

TfidfVectorizer(...):

This class is part of the sklearn.feature_extraction.text module and is used to convert a collection of raw documents (text data) into a matrix of TF-IDF features.

use_idf=True: This parameter indicates that the Inverse Document Frequency (IDF) component of the TF-IDF score should be used in the transformation. IDF helps to weigh down the importance of commonly occurring words and gives more weight to rare words.

ngram_range=(1, 2): This parameter specifies the range of n-grams to be extracted. In this case, it will extract both unigrams (1-word sequences) and bigrams (2-word sequences) from the text. This can help capture more context and relationships between words in the documents.

fit_transform(X_train):

This method fits the vectorizer to the training data X_train and transforms the text data into a TF-IDF feature matrix. The resulting X_train_tfvec will be a sparse matrix where each row corresponds to a document in X_train, and each column corresponds to a feature (word or n-gram) from the text data. The values in the matrix represent the TF-IDF scores of the corresponding words/n-grams in each document.

transform(X_test):

This method transforms the test data X_test into a TF-IDF feature matrix using the vocabulary learned from the training data (i.e., it applies the same transformation to the test set). The resulting X_test_tfvec will also be a sparse matrix similar to X_train_tfvec, but it will only contain the features that were identified during the fit_transform on the training data. This is important to prevent data leakage, ensuring that the model does not have access to information from the test set during training.

Purpose The purpose of this code is to prepare the text data for training a machine learning model by converting the textual information into a numerical format that can be fed into the model. TF-IDF is a popular method for representing text data, as it takes into account both the frequency of words in a document and their importance across the entire corpus.

#### I) Training and Evaluation using different Models after augmentation

##### 1) Logistic Regression Model

alr = LogisticRegression(class_weight='balanced') Initializes the Logistic Regression Model

alr.fit(X_train_tfvec,y_train) Fits the model on the training data

y_pred_alr = alr.predict(X_test_tfvec) Obtains preditctions on the test data

accuracy_score(y_test,y_pred_alr) Using accuracy_score() we check the accuracy on the testing dataset

##### confusion matrix

cmalr = confusion_matrix(y_test, y_pred_alr, labels=alr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmalr, display_labels=alr.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "LR_Confusion_Matrix.png"))
plt.show()

cmalr = confusion_matrix(y_test, y_pred_alr, labels=alr.classes_) 
This computes the confusion matrix for your logistic regression model (alr). It compares the true labels (y_test) with the predicted labels (y_pred_alr).

disp = ConfusionMatrixDisplay(confusion_matrix=cmalr, display_labels=alr.classes_) This creates a display object for the confusion matrix (disp), which will be plotted later. The matrix will have labels corresponding to the classes in your model.

disp.plot() This generates a plot of the confusion matrix.

plt.savefig(os.path.join(visuals_folder_aa, "LR_Confusion_Matrix.png")) Saves the confusion matrix plot as an image (LR_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

plt.show() This shows the confusion matrix plot in a window or notebook, depending on your environment.

##### Classification Report and Score Card Update

print(classification_report(y_pred_alr,y_test)) evaluates the performance of logistic regression model on test data using Classification Report

update_score_card(y_test,y_pred_alr,'Initial_Augmented_LR_model') calls the update score card method to update the score card with this model's score

##### Performing Hyperparameter tuning

param_gridalr = { 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'newton-cg', 'saga'] } defines the hyperparameter grid dictionary

grid_searchalr = GridSearchCV(alr, param_gridalr, cv=5, scoring='accuracy') 

This creates a GridSearchCV object that will: 
Perform cross-validation with cv=5 (5-fold cross-validation) to evaluate the model’s performance. Optimize accuracy using scoring='accuracy'. 
Search over the parameter grid param_gridalr to find the best combination of C and solver for the Logistic Regression model (alr).

grid_searchalr.fit(X_train_tfvec, y_train) Fits the grid on the training data

best_alr = grid_searchalr.best_estimator_ : Obtains the Best model and stores it in best_alr

best_alr - checks the best model 

best_alr.fit(X_train_tfvec, y_train) - Fits the best model obtained on the training data 

y_pred_balr = best_alr.predict(X_test_tfvec) - Obtains preditctions on the test data

accuracy_score(y_test,y_pred_balr) : Using accuracy_score() we check the accuracy on the testing dataset

#### Confusion Matrix

cmbalr = confusion_matrix(y_test, y_pred_balr, labels=best_alr.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cmbalr, display_labels=best_alr.classes_)
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "Tuned_LR_Confusion_Matrix.png"))
plt.show()

This code snippet plots a confusion matrix for the tuned LR Model. The code does the same thing as the previous confusion matrix code except that it Saves the confusion matrix plot as an image (Tuned_LR_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

##### Classification Report and Score Card Update

print(classification_report(y_pred_balr,y_test)) evaluates the performance of the tuned logistic regression model on test data using Classification Report

update_score_card(y_test,y_pred_balr,'BestTuned_Augmented_LR_model') calls the update score card method to update the score card with this model's score

##### 2) Random Forest Classifier Model

arf = RandomForestClassifier(n_estimators=200, random_state=0) Initializes the Random Forest Classifier Model

arf.fit(X_train_tfvec,y_train) Fits it on the training data

y_pred_arf = arf.predict(X_test_tfvec) Obtains predictions on the test data

accuracy_score(y_test,y_pred_arf) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmarf = confusion_matrix(y_test, y_pred_arf, labels=arf.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cmarf, display_labels=arf.classes_) 
disp.plot() 
plt.savefig(os.path.join(visuals_folder_aa, "RFC_Confusion_Matrix.png")) 
plt.show()

This code snippet plots a confusion matrix for the RFC Model. The code does the same thing as the previous confusion matrix code except that it Saves the confusion matrix plot as an image (RFC_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

##### Classification Report and Score Card Update

print(classification_report(y_pred_arf,y_test)) evaluates the performance of the random forest classifier model on test data using classification report

update_score_card(y_test,y_pred_arf,'Initial_Augmented_RFC_model') calling the update score card method to update the score card with this model's score

##### 3) Support Vector Machine Classifier Model

aSVC = LinearSVC() Initializes the Support Vector Machine Classifier

aSVC.fit(X_train_tfvec,y_train) Fits the model on the training data

y_pred_aSVC = aSVC.predict(X_test_tfvec) Obtains the predictions on the test data

accuracy_score(y_test,y_pred_aSVC) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmasvc = confusion_matrix(y_test, y_pred_aSVC, labels=aSVC.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cmasvc, display_labels=aSVC.classes_) 
disp.plot() 
plt.savefig(os.path.join(visuals_folder_aa, "SVC_Confusion_Matrix.png")) 
plt.show()

This code snippet plots a confusion matrix for the SVC Model. The code does the same thing as the previous confusion matrix code except that it Saves the confusion matrix plot as an image (SVC_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

##### Classification Report and Score Card Update

print(classification_report(y_pred_aSVC,y_test)) evaluates the performance of the Support Vector Classifier on test data using Classification Report

update_score_card(y_test,y_pred_aSVC,'Initial_Augmented_SVC_model') calls the update score card method to update the score card with this model's score

##### Performing Hyperparameter tuning

param_grid_asvc = { 'C': [0.01, 0.1, 1, 10, 100] } defines the hyperparameter grid dictionary

grid_search_asvc = GridSearchCV(aSVC, param_grid_asvc, cv=5, scoring='accuracy')

This creates a GridSearchCV object that will: 
Perform 5-fold cross-validation (cv=5) to evaluate the Support Vector Classifier (SVC) model's performance. 
Optimize based on accuracy using scoring='accuracy'. 
Search through the hyperparameter grid (param_grid_asvc) to find the best combination of parameters for the SVC model.

grid_search_asvc.fit(X_train_tfvec, y_train) Fits the grid on the training data

best_asvc = grid_search_asvc.best_estimator_ 
print(f"Best parameters: {grid_search_asvc.best_params_}") 
print(f"Best cross-validation accuracy: {grid_search_asvc.best_score_}")

This snippet Gets and prints the best model and parameters

best_asvc - checks the best model 

best_asvc.fit(X_train_tfvec, y_train) - Fits the best model obtained on the training data

y_pred_basvc = best_asvc.predict(X_test_tfvec) - Obtains the predictions on the test data

accuracy_score(y_test,y_pred_basvc) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion Matrix

cmbasvc = confusion_matrix(y_test, y_pred_basvc, labels=best_asvc.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cmbasvc, display_labels=best_asvc.classes_) 
disp.plot() 
plt.savefig(os.path.join(visuals_folder_aa, "Tuned_SVC_Confusion_Matrix.png")) 
plt.show()

This code snippet plots a confusion matrix for the Tuned SVC Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (Tuned_SVC_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

##### Classification Report and Score Card Update

print(classification_report(y_pred_basvc,y_test)) evaluate the performance of the tuned Support Vector Classifier model on test data using Classification Report

update_score_card(y_test,y_pred_basvc,'BestTuned_Augmented_SVC_model') calls the update score card method to update the score card with this model's score

#### J) Deep Learning Approach¶

##### Importing tensorflow and keras libraries for Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

##### A] LSTM (Long Short Term Memory) Model

##### Converting text data to numeric format and padding

```
Performing Tokenization on a column of text data using the Keras Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

num_words=10000: Limits the tokenizer to the top 10,000 most frequent words in the dataset.
oov_token="<OOV>": Specifies a token ("<OOV>") to represent out-of-vocabulary words—words 
that were not in the top 10,000 during training.
    
```

tokenizer.fit_on_texts(final_df['clean_content'])
This method processes the text data passed (final_df['clean_content']) and builds a vocabulary index based on word frequency.

After this step, the tokenizer will have a dictionary mapping each word in the vocabulary to an integer index.

X_sequences = tokenizer.texts_to_sequences(final_df['clean_content'])

This line converts the text data into a sequence of integers.
The tokenizer is a Tokenizer object from Keras, which has been trained to convert words to integer indices (usually based on word frequency).
texts_to_sequences() takes the cleaned text from the column clean_content of final_df and maps each word in the text 
to its corresponding index in the tokenizer’s vocabulary.

The result, X_sequences, is a list of lists, where each sublist contains the integer sequence corresponding to a 
particular review or document in final_df.
    
X_padded = pad_sequences(X_sequences, maxlen=100, padding='post', truncating='post')

This line ensures that all the sequences have the same length.
pad_sequences() pads or truncates each sequence in X_sequences to a fixed length of 100.
The argument padding='post' adds zeros at the end of sequences that are shorter than 100.
The argument truncating='post' truncates sequences that are longer than 100 from the end.
The result, X_padded, is a 2D array where each row corresponds to a padded sequence of length 100.

This code is converting text data into a numerical format (sequences of word indices) and ensuring all sequences
are of the same length by padding or truncating as needed.
    
##### Label Encoding

from sklearn.preprocessing import LabelEncoder - imports LabelEncoder

Convert labels to categorical format if necessary
label_encoder = LabelEncoder() initializes an instance of the LabelEncoder class from Scikit-learn.

LabelEncoder is used to convert categorical labels into a numerical format.
It is often used when you have target labels (i.e., the y variable) that are 
categorical (e.g., "positive," "neutral," "negative") and you need to convert them into numerical values (e.g., 0, 1, 2) 
for machine learning models.
    
y = label_encoder.fit_transform(final_df['intensity'])

The fit_transform() method does two things:
Fit: It learns the unique classes from the intensity column and assigns each class a unique integer.
Transform: It converts each category in the intensity column into its corresponding integer representation.

The encoded labels are assigned to the variable y. This will be a NumPy array where each entry is an integer 
representing the corresponding intensity label.
    
X_padded.shape - checks the shape of X_padded
    
X_padded - checks X-padded
    
y.shape - checks the shape of y
    
y - checks y

##### Creating a new dataframe combining X_padded & y, saving it
    
lstm_df = pd.DataFrame(np.column_stack((X_padded, y)))
Creates a new dataframe by combining X_padded & y
    
The line lstm_df = pd.DataFrame(np.column_stack((X_padded, y))) is used to create a new pandas DataFrame, 
lstm_df, by combining the X_padded data (your padded sequences) and the y data (your encoded target labels). 

np.column_stack((X_padded, y)):
This function from NumPy horizontally stacks arrays or sequences of arrays.
X_padded is a 2D array of the padded sequences (with shape (num_samples, 100)).
y is the array of labels (with shape (num_samples,)).
np.column_stack combines them into a new 2D array where y becomes an additional column, resulting in a shape of 
(num_samples, 101). Each row will contain 100 values from X_padded followed by the corresponding y label.

pd.DataFrame():

This converts the resulting NumPy array into a pandas DataFrame.
Each row in lstm_df will contain the padded sequence for a specific text review (from X_padded) and the 
corresponding intensity label (from y).
    
lstm_df.head() - checks lstm_df
    
file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "lstm_data.csv")
Specifies the file path where you want to save the file
    
lstm_df.to_csv(file_path, index=False)  
Saves the lstm_df dataframe as a CSV file
Sets index=False to avoid saving the DataFrame index
    
##### Performing train test Split for LSTM¶
    
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    
splits the data into training and test set
    
##### Build LSTM Model
    
lstm_model1 = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes
])

This code snippet defines a Long Short-Term Memory (LSTM) model using Keras' Sequential API for text classification

i) Sequential():

This creates a linear stack of layers for your neural network. Each layer feeds into the next.

ii) Embedding(input_dim=10000, output_dim=128, input_length=100):

This layer converts integer-encoded words into dense vectors of fixed size (in this case, 128).

input_dim=10000: The size of the vocabulary (the maximum number of unique tokens/words). 
Here, it means the model can handle up to 10,000 different words.

output_dim=128: The dimensionality of the dense embedding output for each word.

input_length=100: This specifies that each input sequence has a length of 100.

iii) Bidirectional(LSTM(64, return_sequences=False)):

This adds a Bidirectional LSTM layer with 64 units.

Bidirectional: This allows the model to learn patterns in both directions (forward and backward) in the input sequence, 
which can be beneficial for understanding context.

LSTM(64): Specifies that the LSTM layer will have 64 memory units (or hidden states).

return_sequences=False: This means that the LSTM will only return the output of the last time step 
(the final output of the sequence) instead of returning the output for every time step. 

iv) Dropout(0.5):

This layer helps prevent overfitting by randomly setting 50% of the inputs to zero during training, 
which encourages the model to learn more robust features.

v) Dense(64, activation='relu'):

A fully connected (dense) layer with 64 units and ReLU (Rectified Linear Unit) activation function. 
This layer can help the model learn complex representations.

vi) Dense(3, activation='softmax'):

This is the output layer with 3 units (for three classes) and a softmax activation function.

activation='softmax': This ensures that the output is a probability distribution over the three classes, 
where the sum of the probabilities equals 1. Each output corresponds to the likelihood of the input belonging to each class.

##### Compile the LSTM Model
    
lstm_model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

This line is used to configure the LSTM model for training

loss='sparse_categorical_crossentropy':

This specifies the loss function to be used during training.
sparse_categorical_crossentropy is appropriate for multi-class classification problems where the target labels are integers.

This loss function computes the cross-entropy loss between the predicted probability distribution 
(from the softmax output layer) and the true labels, which helps the model learn by minimizing this loss.

optimizer='adam':

This specifies the optimization algorithm used for training the model.
adam (Adaptive Moment Estimation) is a popular optimizer that adjusts the learning rate for each parameter
based on first and second moments of the gradients. It generally works well in practice and often leads to faster convergence.

metrics=['accuracy']:

This specifies the metrics to evaluate the model's performance during training and testing.
accuracy is a common metric for classification tasks, measuring the ratio of correctly predicted samples to the total samples.
    
    
lstm_model1 - checks the lstm_model1
    
lstm_model1.summary() - this generates a summary of the LSTM model architecture, providing an overview of the layers, 
their output shapes, and the number of parameters in the model

##### Training the LSTM Model    

history = lstm_model1.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

this line of code trains the LSTM model on the training data for 10 epochs, using a batch size of 32, 
while also evaluating its performance on a separate validation dataset. 

The training history will help you analyze the model’s performance and potentially fine-tune hyperparameters or 
architecture later.
    
lstm_model1.fit(...):

This method trains the model using the provided training data.

X_train:

This is the input data for training, which should be a 2D array of shape (num_samples, 100), where each row corresponds to a padded sequence of text data.

y_train:

This is the target data for training, which should be a 1D array containing the encoded class labels corresponding to each sample in X_train.

epochs=10:

This specifies the number of times the entire training dataset will be passed forward and backward through the model. In this case, the model will train for 10 epochs.

validation_data=(X_test, y_test):

This argument provides validation data to evaluate the model’s performance after each epoch.

X_test is the input data for validation, and y_test is the corresponding target labels. This allows you to monitor the model's performance on unseen data during training.

batch_size=32:

This defines the number of samples processed before the model is updated. In this case, the model will update its weights after processing every 32 samples. Smaller batch sizes can lead to more noisy updates, while larger batch sizes can smooth out the updates but may require more memory.

history =:

This stores the output of the fit() method, which contains the training and validation metrics for each epoch, such as loss and accuracy. You can use this history object to visualize the training process and evaluate the model's performance over time.

lstm_model1.summary() - Prints the summary of the lstm_model1 after fittting on the training data

##### Evaluating the model
    
test_loss, test_acc = lstm_model1.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}") - This line prints the test accuracy to the console
    
lstm_model1.evaluate(X_test, y_test):

The evaluate method calculates the loss and any specified metrics (like accuracy) for the model using the provided test data. 

X_test: This is the input data for testing, which should be a 2D array of padded sequences. 
y_test: This is the corresponding target labels for the test data, which contains the encoded class labels.

test_loss, test_acc =:

The evaluate method returns two values:

test_loss: This is the value of the loss function calculated on the test data. It indicates how well the model performs in terms of its objective (in this case, minimizing sparse categorical cross-entropy).

test_acc: This is the accuracy of the model on the test data, showing the proportion of correctly predicted samples out of all test samples.

y_pred_lstm = lstm_model1.predict(X_test)
generate predictions from the trained LSTM model for the test dataset.
    
y_pred_lstm - checks the predictions
    
y_pred_lstm_labels = np.argmax(y_pred_lstm, axis=1) - Converts the predicted probabilities to class labels
    
np.argmax(y_pred_lstm, axis=1):

The np.argmax() function from the NumPy library returns the indices of the maximum values along the specified axis.

y_pred_lstm: This is the 2D array of predicted probabilities generated by the LSTM model for each class. 
axis=1: This specifies that the operation should be performed along the rows. In this case, it means that for each row (each sample), it will find the index of the highest probability across the columns (classes).

y_pred_lstm_labels =:

The result of the np.argmax() operation is assigned to the variable y_pred_lstm_labels. This will be a 1D array containing the class labels corresponding to the highest predicted probabilities for each sample in the test set.

y_pred_lstm_labels - checks the class labels generated
    
accuracy_score(y_test,y_pred_lstm_labels) - Using accuracy_score() to check the accuracy on the testing dataset
    
np.unique(y_test) - checks the unique labels in y_test
    
##### Confusion Matrix¶
    
cmlstm = confusion_matrix(y_test,y_pred_lstm_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmlstm, display_labels=np.unique(y_test))
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "LSTM_Confusion_Matrix.png"))
plt.show()
    
This code snippet plots a confusion matrix for the LSTM Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (LSTM_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).
    
##### Classification Report and Score Card Update
    
print(classification_report(y_test,y_pred_lstm_labels)) - evaluate the performance of the LSTM model on test data using Classification Report
    
update_score_card(y_test,y_pred_lstm_labels,'Initial_LSTM_model') - calls the update score card method to update the score card with this model's score
    
##### Performing Hyperparameter tuning
    
pip install keras-tuner - installs keras-tuner which needs to be installed for hyperparameter tuning
    
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
importing libraries required for hyperparameter tuning
    
##### Steps involved

1) defining the hypermodel
The function build_model is defined to create a Sequential model. It accepts a hp argument, which represents the hyperparameter space.

Embedding Layer: The output dimension of the embedding layer is defined as a hyperparameter that can vary between 64 and 256 with a step of 32.

LSTM Layer: The number of LSTM units is also defined as a hyperparameter between 32 and 256.

Dropout Layer: The dropout rate is defined as a hyperparameter ranging from 0.2 to 0.5, which helps prevent overfitting.

Dense Layer: The number of units in the dense layer is another hyperparameter, ranging from 32 to 128.

Output Layer: The final layer uses a softmax activation function for multi-class classification (3 classes).

Model Compilation: The model is compiled using Adam optimizer with a learning rate defined as a hyperparameter, sparse categorical cross-entropy loss, and accuracy as a metric.

2) Instantiating the Tuner
Tuner initialization: A RandomSearch tuner is instantiated, which will search for the best hyperparameters.

Parameters:

objective='val_accuracy': The tuner aims to maximize validation accuracy.

max_trials=5: The tuner will try 5 different combinations of hyperparameters.

executions_per_trial=1: For each trial, one model will be built and evaluated.

directory and project_name: These specify where to save the tuning results.

3) Search for the Best Hyperparameters
Hyperparameter Search: The search method runs the random search to find the best hyperparameters based on the training and validation data. It trains each model for 10 epochs using a batch size of 32.

4) Get the Best Model
Retrieve Best Hyperparameters: The best hyperparameters found during the search are stored in best_hps.

Build Best Model: The best model is created using the hyperparameters retrieved.

5) Train the Best Model
Training: The best-tuned model is trained on the training dataset for 10 epochs, using the same validation data.

6) Evaluate the Best Model
Model Evaluation: The model is evaluated on the test dataset, returning the test loss and accuracy.

Print Accuracy: The accuracy of the best model on the test data is printed
    
-------------------------------------------------------------------------------------------------------------------------------
    
Define the hypermodel
def build_model(hp):
    lstm_tuned_model = Sequential()

    Embedding layer
    lstm_tuned_model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int('embedding_output', min_value=64, max_value=256, step=32),
                        input_length=100))

    LSTM layer
    lstm_tuned_model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=256, step=32), return_sequences=False)))

    Dropout layer
    lstm_tuned_model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    Dense layer
    lstm_tuned_model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))

    Output layer
    lstm_tuned_model.add(Dense(3, activation='softmax'))  # 3 output classes

    Compile the model
    lstm_tuned_model.compile(optimizer=tf.keras.optimizers.Adam(
                      learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return lstm_tuned_model

Instantiate the tuner
lstmtuner1 = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to build and evaluate for each trial
    directory='lstm_tuning',
    project_name='text_intensity_lstm')

Search for the best hyperparameters
lstmtuner1.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

Get the best model
best_hps = lstmtuner1.get_best_hyperparameters()[0]
best_tuned_lstm_model = lstmtuner1.hypermodel.build(best_hps)

Train the best model
history = best_tuned_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

Evaluate the best model
test_loss, test_acc = best_tuned_lstm_model.evaluate(X_test, y_test)
print(f"Best Model Test Accuracy: {test_acc}")
    
-------------------------------------------------------------------------------------------------------------------------------
##### Checking and Evaluating the best model

best_tuned_lstm_model - checks the best tuned model
    
best_tuned_lstm_model.summary() - Prints the summary of the best_tuned_lstm_model after fittting on the training data
    
y_pred_btlstm = best_tuned_lstm_model.predict(X_test) - generate predictions from the best tuned LSTM model for the test dataset.
    
y_pred_btlstm_labels = np.argmax(y_pred_btlstm, axis=1) - Convert the predicted probabilities to class labels
    
accuracy_score(y_test,y_pred_btlstm_labels) - Using accuracy_score() we check the accuracy on the testing dataset

##### Confusion matrix
    
cmbtlstm = confusion_matrix(y_test,y_pred_btlstm_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmbtlstm, display_labels=np.unique(y_test))
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "Tuned_LSTM_Confusion_Matrix.png"))
plt.show()
    
This code snippet plots a confusion matrix for the Tuned LSTM Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (Tuned_LSTM_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).
    
##### Classification Report and Score Card Update
    
print(classification_report(y_test,y_pred_btlstm_labels))
evaluates the performance of the Tuned LSTM model on test data using Classification Report
    
update_score_card(y_test,y_pred_btlstm_labels,'Tuned_LSTM_model')
calls the update score card method to update the score card with this model's score
    
##### Building a hypermodel with a deeper architure by adding another bi-directional layer
    
Key Changes made
Multiple LSTM Layers:

Added a second LSTM layer with return_sequences=False to stop sequence propagation after the second layer.

Hyperparameter Tuning for Multiple Layers:

Hyperparameters are now tuned for both LSTM layers (lstm_units_1 and lstm_units_2). Separate dropout rates for each layer (dropout_rate_1, dropout_rate_2), including a dense layer dropout (dropout_rate_dense).

Additional Dense Layer:

A deeper Dense layer before the output has been added to introduce more non-linearity to the model.
    
-------------------------------------------------------------------------------------------------------------------------
    
Define the hypermodel with deeper architecture
def build_model(hp):
    deeper_lstm_model = Sequential()

    Embedding layer
    deeper_lstm_model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int('embedding_output', min_value=64, max_value=256, step=32),
                        input_length=100))

    First LSTM layer (stacked)
    deeper_lstm_model.add(Bidirectional(LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=256, step=32), return_sequences=True)))
    deeper_lstm_model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))

    Second LSTM layer (stacked)
    deeper_lstm_model.add(Bidirectional(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=256, step=32), return_sequences=False)))
    deeper_lstm_model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.2, max_value=0.5, step=0.1)))

    Dense layer
    deeper_lstm_model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'))
    deeper_lstm_model.add(Dropout(rate=hp.Float('dropout_rate_dense', min_value=0.2, max_value=0.5, step=0.1)))

    Output layer (3 output classes)
    deeper_lstm_model.add(Dense(3, activation='softmax'))  # For 3 classes: 'angriness', 'happiness', 'sadness'

    Compile the model
    deeper_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(
                      learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return deeper_lstm_model

Instantiate the tuner
lstmtuner2 = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to build and evaluate for each trial
    directory='lstm_tuning',
    project_name='text_intensity_lstm')

Search for the best hyperparameters
lstmtuner2.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

Get the best model
best_hps = lstmtuner2.get_best_hyperparameters()[0]
best_deep_lstm_model = lstmtuner2.hypermodel.build(best_hps)

Train the best model
history = best_deep_lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

Evaluate the best model
test_loss, test_acc = best_deep_lstm_model.evaluate(X_test, y_test)
print(f"Best Model Test Accuracy: {test_acc}")
    
-----------------------------------------------------------------------------------------------------------------------------
##### Checking and Evaluating the best deep model

best_deep_lstm_model - checks the best deeper tuned LSTM Model
    
best_deep_lstm_model.summary() - Prints the summary of the best_deep_tuned_lstm_model after fittting on the training data
    
y_pred_dtlstm = best_deep_lstm_model.predict(X_test) - generates predictions from the best deeper tuned LSTM model for the test dataset.
    
y_pred_dtlstm_labels = np.argmax(y_pred_dtlstm, axis=1) - Converts the predicted probabilities to class labels
    
accuracy_score(y_test,y_pred_dtlstm_labels) - Using accuracy_score() we are check the accuracy on the testing dataset
    
##### Confusion Matrix
    
cmdtlstm = confusion_matrix(y_test,y_pred_dtlstm_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmdtlstm, display_labels=np.unique(y_test))
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "Deeper_Tuned_LSTM_Confusion_Matrix.png"))
plt.show()
    
This code snippet plots a confusion matrix for the Deeper Tuned LSTM Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (Deeper_Tuned_LSTM_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).
    
##### Classification Report and Score Card Update
    
print(classification_report(y_test,y_pred_dtlstm_labels)) 
evaluates the performance of the Deeper Tuned LSTM model on test data using Classification Report

update_score_card(y_test,y_pred_dtlstm_labels,'Deeper_Tuned_LSTM_model')
calls the update score card method to update the score card with this model's score
    
##### B] CNN (Convolutional Neural Network Model)
    
importing libraries required for CNN

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
    
Defining constants
MAX_VOCAB_SIZE = 10000  - Vocabulary size
MAX_SEQUENCE_LENGTH = 100  - Maximum length of sequences
EMBEDDING_DIM = 100  - Dimensionality of the embedding layer
NUM_CLASSES = 3   -Adjust according to your number of classes

##### Creating the CNN dataset and saving it
    
Extracting texts and labels from final_df
texts = final_df['clean_content'].values
labels = final_df['intensity'].values
    
cnn_df = pd.DataFrame(np.column_stack((texts, labels)), columns=['CleanContent', 'Intensity'])
Concatenates/Stacks texts and labels to create a dataframe for CNN Model
    
file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "cnn_data.csv")
Specifies the file path where you want to save the file

cnn_df.to_csv(file_path, index=False) 
Saves the cnn_df dataframe as a CSV file
Sets index=False to avoid saving the DataFrame index

##### Performing Train Test Split for CNN
    
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
Splits the data into training and testing sets

##### Converting text data to numeric format and padding
    
Performing Tokenization on a column of text data using the Keras Tokenizer
    
tokenizer = Tokenizer(num_words=10000)  # Set the maximum vocabulary size
tokenizer.fit_on_texts(X_train)

This method processes the text data passed (X_train) and builds a vocabulary index based on word frequency.
After this step, the tokenizer will have a dictionary mapping each word in the vocabulary to an integer index.
    
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
Converts text to sequences
    
Padding the sequences

MAX_SEQUENCE_LENGTH = 100  # Set your desired max length
X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH)
X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH)

Since the input sequences need to be of the same length, you use pad_sequences to pad (or truncate) them to a fixed length, 
defined by MAX_SEQUENCE_LENGTH. In this case, the sequences are either truncated or padded to a length of 100.
    
##### Label Encoding

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

You use LabelEncoder to convert the categorical labels (y_train and y_test) into integer-encoded values. 
The fit_transform method is used on the training labels, while transform is used on the test labels to ensure 
consistent encoding.

The texts_to_sequences method converts each sentence in the training and testing datasets into a sequence of integers, 
where each integer represents a word in the tokenizer's vocabulary.
    
cnn_df - checks cnn_df
    
print("Training data shape:", X_train_padded.shape, y_train_encoded.shape)
print("Testing data shape:", X_test_padded.shape, y_test_encoded.shape)
    
checks and prints the shape of X_train_padded, y_train_encoded, X_test_padded, y_test_encoded
    
-----------------------------------------------------------------------------------------------------------------------------
    
##### Defining a Function to Build and Compile the CNN Model
    
def create_cnn_model(vocabulary_size, embedding_dim, max_length):
    cnnmodel = Sequential()

    Embedding layer
    cnnmodel.add(Embedding(input_dim=vocabulary_size,
                        output_dim=embedding_dim,
                        input_length=max_length))

    Convolutional layer
    cnnmodel.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

    Max pooling layer
    cnnmodel.add(MaxPooling1D(pool_size=2))

    Flatten layer
    cnnmodel.add(Flatten())

    Fully connected layer
    cnnmodel.add(Dense(128, activation='relu'))
    cnnmodel.add(Dropout(0.5))

    Output layer
    cnnmodel.add(Dense(3, activation='softmax'))  # Change '3' to the number of classes in your dataset

    Compile the model
    cnnmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return cnnmodel
    
-----------------------------------------------------------------------------------------------------------------------
##### function explanation

This code defines a function create_cnn_model that builds and compiles a Convolutional Neural Network (CNN) model using the Keras Sequential API for text classification tasks. 
 
def create_cnn_model(vocabulary_size, embedding_dim, max_length):
The function accepts three parameters:

vocabulary_size: The size of the vocabulary (number of unique words) for embedding.
embedding_dim: The dimensionality of the dense word vectors (i.e., how each word will be represented in a vector space).
max_length: The maximum length of input sequences (used to pad/truncate sequences to this length).
    
cnnmodel = Sequential()
A Sequential model allows you to build a neural network layer by layer, where each layer has one input tensor and one output tensor.
    
cnnmodel.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length))

Embedding Layer: Converts words into dense vectors of fixed size (embedding_dim).
input_dim=vocabulary_size: The size of the vocabulary (number of words).
output_dim=embedding_dim: The size of each embedding vector.
input_length=max_length: Each input sentence will be padded/truncated to this length.
    
cnnmodel.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    
Conv1D Layer: Applies 1D convolutions over the input sequences (useful for detecting local patterns in text).
filters=128: The number of output filters in the convolution.
kernel_size=5: The size of the convolutional window (i.e., looks at 5 adjacent words at a time).
activation='relu': The ReLU activation function is applied, introducing non-linearity.
    
cnnmodel.add(MaxPooling1D(pool_size=2))
    
MaxPooling1D Layer: Reduces the dimensionality by selecting the maximum value over a window (helps retain important features while reducing computation).
pool_size=2: The size of the pooling window (i.e., reduces the sequence length by half).
    
cnnmodel.add(Flatten())
Flatten Layer: Converts the 3D output of the convolutional and pooling layers into a 1D vector (necessary for feeding into fully connected layers).
    
cnnmodel.add(Dense(128, activation='relu'))
cnnmodel.add(Dropout(0.5))

Dense Layer: A fully connected layer with 128 neurons and ReLU activation.
Dropout Layer: Adds a dropout rate of 50% to prevent overfitting (randomly drops neurons during training).
    
cnnmodel.add(Dense(3, activation='softmax'))
    
Dense Layer: The output layer with 3 neurons (assuming this is a 3-class classification problem).
Softmax Activation: Used for multi-class classification, converts the output into a probability distribution.
    
cnnmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Optimizer: Adam optimizer is used, which combines the advantages of RMSProp and Stochastic Gradient Descent.
Loss: sparse_categorical_crossentropy is used for multi-class classification where the labels are integer-encoded (not one-hot encoded).
Metrics: The model tracks accuracy during training and evaluation.
    
return cnnmodel
The function returns the compiled CNN model.

This CNN is designed for text data, where the embedding layer maps words to dense vectors, followed by a convolutional layer to capture patterns in the word sequences, and finally fully connected layers for classification.

##### Creating and Training the CNN Model by calling the above function

Defining constants
VOCABULARY_SIZE = 10000  Based on the tokenizer
EMBEDDING_DIM = 128      You can choose the embedding dimension
MAX_LENGTH = 100         Same as the max length used for padding
    
cnn_model = create_cnn_model(VOCABULARY_SIZE, EMBEDDING_DIM, MAX_LENGTH) - Creates the model by calling create_cnn_model by passing the constants defined as arguments

Training the model
history = cnn_model.fit(X_train_padded, y_train_encoded,
                         epochs=10, batch_size=32,
                         validation_split=0.2)
    
This piece of code is training the CNN model (cnn_model) using the .fit() method. Here's a breakdown:

Explanation of Each Argument:

X_train_padded: This is your input data for training. It likely contains padded sequences of text, with each sequence having a fixed length (max_length), as required by the CNN.

y_train_encoded: This is the corresponding target/output labels for your training data, typically encoded as integers (for classification problems).

Hyperparameters:
epochs=10: The model will train for 10 complete passes through the entire training dataset. One epoch means training on the entire training set once.

batch_size=32: The training data will be split into mini-batches of 32 samples each. The model will update its weights after each batch, making the training more efficient.

Validation:
validation_split=0.2: This indicates that 20% of the training data will be set aside as a validation set. During training, the model will evaluate its performance on this data (which is not used for training) to monitor overfitting and generalization to unseen data.

Output:
history: The result of the .fit() method is stored in the history object. This object contains training details, including loss and accuracy over epochs for both the training and validation sets. You can use this history to visualize the learning curves (e.g., loss and accuracy over time).

In summary, this code trains the cnn_model on X_train_padded and y_train_encoded data for 10 epochs, with a batch size of 32, while also validating the model performance on 20% of the training data.
    
##### Evaluating the CNN Model

test_loss, test_acc = cnn_model.evaluate(X_test_padded, y_test_encoded)
print(f"Test Accuracy: {test_acc:.2f}")
- evaluates the model and prints the accuracy
    
cnn_model - Checks the CNN Model
    
cnn_model.summary() - Prints the summary of the CNN model after fittting on the training data
    
y_pred_cnn = cnn_model.predict(X_test_padded) - generates predictions from the CNN model for the test dataset.
    
y_pred_cnn_labels = np.argmax(y_pred_cnn, axis=1) - Converts the predicted probabilities to class labels
    
accuracy_score(y_test_encoded,y_pred_cnn_labels) - Using accuracy_score() we check the accuracy on the testing dataset
    
##### confusion matrix
    
cmcnn = confusion_matrix(y_test_encoded,y_pred_cnn_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmcnn, display_labels=np.unique(y_test_encoded))
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "CNN_Confusion_Matrix.png"))
plt.show()
    
This code snippet plots a confusion matrix for the CNN Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (CNN_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).
    
##### Classification Report and Score Card Update
    
print(classification_report(y_test_encoded,y_pred_cnn_labels)) - evaluates the performance of the CNN model on test data using Classification Report
    
update_score_card(y_test_encoded,y_pred_cnn_labels,'Initial_CNN_model') - calls the update score card method to update the score card with this model's score
    
##### Performing Hyperparameter Tuning
    
---------------------------------------------------------------------------------------------------------------------------
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding, Dropout

def build_model(hp):
    tcnnmodel = Sequential()
    tcnnmodel.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH))
    tcnnmodel.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=256, step=32), kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]), activation='relu'))
    tcnnmodel.add(MaxPooling1D(pool_size=2))
    tcnnmodel.add(Flatten())
    tcnnmodel.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    tcnnmodel.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)))
    tcnnmodel.add(Dense(NUM_CLASSES, activation='softmax'))
    tcnnmodel.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return tcnnmodel

cnntuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='cnn_hyperparam_tuning')

cnntuner.search(X_train_padded, y_train_encoded, epochs=10, validation_split=0.2)

best_hps = cnntuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best hyperparameters: {best_hps.values}")

best_tcnnmodel = cnntuner.hypermodel.build(best_hps)
best_tcnnmodel.summary()
    
-------------------------------------------------------------------------------------------------------------------------------
##### function explanation

def build_model(hp):
    tcnnmodel = Sequential()

This function defines a CNN model and uses the hp (hyperparameter) object provided by Keras Tuner to vary the hyperparameters.

Embedding Layer:

tcnnmodel.add(Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH))
The embedding layer converts words into dense vectors, where VOCABULARY_SIZE, EMBEDDING_DIM, and MAX_LENGTH are predefined constants.

Conv1D Layer with tunable hyperparameters:

tcnnmodel.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=256, step=32), kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]), activation='relu'))

The hyperparameters for this layer are:

filters: Tunable number of filters (32 to 256 in steps of 32).
kernel_size: Tunable kernel size, which can be either 3, 5, or 7.

MaxPooling Layer: 
tcnnmodel.add(MaxPooling1D(pool_size=2))
The max pooling layer reduces the dimensionality of the feature maps.

Flatten Layer:
tcnnmodel.add(Flatten())
The output is flattened into a 1D vector to feed into fully connected layers.

Fully Connected (Dense) Layer with tunable units and dropout:

tcnnmodel.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
tcnnmodel.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)))

units: Tunable number of neurons (32 to 512 in steps of 32).
dropout_rate: Tunable dropout rate (0.3 to 0.7 in steps of 0.1).

Output Layer:

tcnnmodel.add(Dense(NUM_CLASSES, activation='softmax'))
A fully connected layer with NUM_CLASSES neurons, using the softmax activation for multi-class classification.

Compilation:

tcnnmodel.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

The optimizer is tunable, either 'adam' or 'rmsprop'.
loss='sparse_categorical_crossentropy': Used for multi-class classification where the labels are integer-encoded.
metrics=['accuracy']: The model tracks accuracy during training.
    
Hyperparameter Tuning: RandomSearch

cnntuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='cnn_hyperparam_tuning')

kt.RandomSearch: A random search strategy for hyperparameter tuning.
build_model: The model-building function where hyperparameters are tuned.
objective='val_accuracy': The goal is to maximize validation accuracy.
max_trials=10: The tuner will try 10 different hyperparameter combinations.
executions_per_trial=1: Each trial (combination of hyperparameters) is run once.
directory='my_dir': The directory where the results of the tuning process will be saved.
project_name='cnn_hyperparam_tuning': The name of the project, used for organizing results.
    
 
Hyperparameter Search

cnntuner.search(X_train_padded, y_train_encoded, epochs=10, validation_split=0.2)

This command starts the hyperparameter search:
X_train_padded and y_train_encoded are the training data and labels.
epochs=10: Each model will be trained for 10 epochs.
validation_split=0.2: 20% of the training data is set aside for validation.

Best Hyperparameters

best_hps = cnntuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

get_best_hyperparameters(): This retrieves the best hyperparameter configuration after the search.
num_trials=1: Retrieves the best combination (in this case, only one is needed).
best_hps.values: Prints the best hyperparameters found.

Building and Summarizing the Best Model
best_tcnnmodel = cnntuner.hypermodel.build(best_hps)
best_tcnnmodel.summary()

best_tcnnmodel: This builds the best model using the optimal hyperparameters (best_hps).
summary(): Prints a summary of the model architecture, including layer types and the number of parameters.

Summary:
This code defines a CNN for text classification and tunes hyperparameters such as the number of filters, kernel size, units in the dense layer, dropout rate, and optimizer.
It uses Keras Tuner's RandomSearch to find the best set of hyperparameters by trying 10 different combinations and selecting the one with the highest validation accuracy.
 
##### Training the best CNN Model obtained
    
history = best_tcnnmodel.fit(X_train_padded, y_train_encoded, 
                             epochs=10, 
                             validation_split=0.2)

This piece of code is training the Best Tuned CNN model (best_tcnnmodel) using the .fit() method. The explanation is the same as for training the initial CNN Model.
    
best_tcnnmodel - checks the best CNN Model
    
best_tcnnmodel.summary() - Prints the summary of the best tuned CNN model after fittting on the training data

y_pred_tcnn = best_tcnnmodel.predict(X_test_padded) - generates predictions from the Tuned CNN model for the test dataset.
    
y_pred_tcnn_labels = np.argmax(y_pred_tcnn, axis=1) - Converts the predicted probabilities to class labels
    
accuracy_score(y_test_encoded,y_pred_tcnn_labels) - Using accuracy_score() we check the accuracy on the testing dataset
    
##### confusion matrix
    
cmtcnn = confusion_matrix(y_test_encoded,y_pred_tcnn_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmtcnn, display_labels=np.unique(y_test_encoded))
disp.plot()
plt.savefig(os.path.join(visuals_folder_aa, "Tuned_CNN_Confusion_Matrix.png"))
plt.show()
    
This code snippet plots a confusion matrix for the Tuned CNN Model. The code does the same thing as the previous confusion matrix code except that it saves the confusion matrix plot as an image (Tuned_CNN_Confusion_Matrix.png) in the specified folder (visuals_folder_aa).

##### Classification Report and Score Card Update

print(classification_report(y_test_encoded,y_pred_tcnn_labels)) - evaluates the performance of the Tuned CNN model on test data using Classification Report
    
update_score_card(y_test_encoded,y_pred_tcnn_labels,'Tuned_CNN_model') - calls the update score card method to update the score card with this model's score
    
#### K) Saving the models built
       
model_folder = "C:/Users/nikde/Documents/UpGrad/intensityanalysis/models"
Defines the folder where models will be saved
    
joblib.dump(lr, os.path.join(model_folder, "Initial_LR_model.pkl"))
joblib.dump(best_lr, os.path.join(model_folder, "BestTuned_LR_model.pkl"))
joblib.dump(rf, os.path.join(model_folder, "Initial_RFC_model.pkl"))
joblib.dump(best_rf, os.path.join(model_folder, "BestTuned_RFC_model.pkl"))
joblib.dump(SVC, os.path.join(model_folder, "Initial_SVC_model.pkl"))
joblib.dump(best_svc, os.path.join(model_folder, "BestTuned_SVC_model.pkl"))

joblib.dump(alr, os.path.join(model_folder, "Initial_Augmented_LR_model.pkl"))
joblib.dump(best_alr, os.path.join(model_folder, "BestTuned_Augmented_LR_model.pkl"))
joblib.dump(arf, os.path.join(model_folder, "Initial_Augmented_RFC_model.pkl"))
joblib.dump(aSVC, os.path.join(model_folder, "Initial_Augmented_SVC_model.pkl"))
joblib.dump(best_asvc, os.path.join(model_folder, "BestTuned_Augmented_SVC_model.pkl"))

joblib.dump(lstm_model1, os.path.join(model_folder, "Initial_LSTM_model.pkl"))
joblib.dump(best_tuned_lstm_model, os.path.join(model_folder, "Tuned_LSTM_model.pkl"))
joblib.dump(best_deep_lstm_model, os.path.join(model_folder, "Deeper_Tuned_LSTM_model.pkl"))

joblib.dump(cnn_model, os.path.join(model_folder, "Initial_CNN_model.pkl"))
joblib.dump(best_tcnnmodel, os.path.join(model_folder, "Tuned_CNN_model.pkl"))

print("All models saved successfully!")
    
This snippet saves each model with a different name in the models folder and prints a success message after saving them.
    
    
#### L) Performance Comparison of all the models and printing the Final Scorecard
    
sorted_score_card = score_card.sort_values(by='Accuracy Score', ascending=False).reset_index(drop=True)  
- Sorts the score card generated by accuracy score in descending order

final_score_card = sorted_score_card.head(10) - Stores only the top 10 models into final_score_card

final_score_card - prints the final score card
    
file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/intensityanalysis/data/processeddata", "best_performing_models.csv")
- Specifies the file path where you want to save the file

final_score_card.to_csv(file_path, index=False)  
Saves the final_score_card dataframe as a CSV file    
Sets index=False to avoid saving the DataFrame index

##### M) Deployment Plan and Future course

After confirming the proper functioning of the project, I have uploaded the project folder with all the necessary files and folders to a newly created repository named (intensityanalysis) on my GitHub profile. 

Here is the link to the project repository : 

##### Deployment on Cloud Platforms

To take this project to the next level, it can be deployed using various cloud platforms that specialize in machine learning services. Some of the most popular platforms include:

AWS SageMaker: SageMaker allows for easy deployment of machine learning models. The platform provides pre-built infrastructure to train, host, and scale models, simplifying the process of deploying a model to a production environment.

Azure Machine Learning: Azure ML provides a managed cloud environment where models can be trained and deployed. It supports integration with other Azure services, making it ideal for creating a complete data pipeline that handles both training and inference.

Google Cloud Vertex AI: Vertex AI offers end-to-end machine learning workflows, enabling users to build, deploy, and scale models with ease. The platform also allows seamless integration with Google Cloud's data services for handling live data streams.

Deploying the model on one of these platforms will enable real-time predictions using live or streaming data. By integrating the project with a cloud-based infrastructure, we can automate the entire machine learning pipeline, from data ingestion and preprocessing to model deployment and inference.

Why Deploy to the Cloud?

Scalability: Cloud platforms provide the ability to scale the model based on usage, automatically adjusting resources to handle increasing traffic or data volume.

Real-Time Predictions: Deploying the model on the cloud allows for continuous predictions, processing new data as it arrives either in real time or at specified intervals.

End-to-End Automation: Cloud deployment enables automation of the entire ML workflow, from data collection and processing to model training, evaluation, and prediction.

Cost Efficiency: Using managed cloud services allows for pay-as-you-go pricing, ensuring that you only pay for the resources you use. Additionally, the infrastructure is managed by the cloud provider, reducing operational overhead.

##### Future Course

After deploying an NLP-based intensity prediction model on the cloud, it can be used in several ways and integrated into real-world systems across various industries. Here are potential future use cases and integration strategies for such a model:

##### 1. Sentiment and Emotion Analysis Platforms
Use Case: Cloud-based NLP models could be integrated into SaaS platforms offering sentiment and emotion analysis services to businesses.

Integration: These models can be accessed via APIs by enterprises looking to analyze customer reviews, social media mentions, and other feedback. Companies could subscribe to these services to gain actionable insights on customer sentiment intensity, enabling them to adjust their marketing or customer service strategies dynamically.

##### 2. Customer Support Automation
Use Case: Integrating intensity prediction into customer support systems can help prioritize critical issues and improve response times for high-intensity (e.g., highly negative) cases.

Integration: Cloud-based services could link the model to ticketing systems (like Zendesk or Freshdesk), where highly intense reviews or complaints are flagged for immediate attention. This ensures customer service teams focus on solving urgent issues, improving efficiency and customer satisfaction.

##### 3. E-commerce and Product Review Filtering
Use Case: E-commerce platforms can use the model to filter out irrelevant or spammy reviews based on extreme intensity levels or detect genuine feedback.

Integration: This model can be incorporated into product review sections of online retailers like Amazon, where it helps classify, sort, and filter reviews. Intense positive or negative reviews can be highlighted to potential buyers, aiding decision-making and improving trust in the review system.

##### 4. Social Media Monitoring and Crisis Management
Use Case: Businesses and political entities can monitor public sentiment across social media and news platforms. Detecting high-intensity comments may help in identifying potential crises.

Integration: The model can be deployed to monitor real-time Twitter, Facebook, or Instagram posts, and provide alerts when intense emotions (e.g., outrage or excitement) are detected, allowing companies to address issues or capitalize on positive trends.

##### 5. Chatbots and Virtual Assistants
Use Case: By integrating intensity prediction, conversational agents like chatbots can adjust their responses based on the emotional intensity of user inputs, offering a more empathetic experience.

Integration: Cloud-hosted intensity prediction models can be plugged into AI-driven customer service chatbots (e.g., through platforms like Dialogflow or Rasa). When the model detects an intense emotional tone, the chatbot can escalate the conversation to a human agent or provide an appropriately sensitive response.

##### 6. Mental Health and Emotional Well-being Applications
Use Case: NLP models could be deployed in mental health applications to analyze user conversations or journal entries, detecting emotionally intense language that may indicate distress.

Integration: Cloud-based models can be embedded into mental health platforms or mobile apps (like Woebot or Youper). The model can flag users who might need immediate intervention or recommend resources based on detected emotional intensity, supporting mental health professionals.

##### 7. HR and Employee Engagement Platforms
Use Case: Companies can use intensity prediction models to monitor employee feedback and engagement, especially in anonymous surveys and feedback forms.

Integration: Deployed through cloud-based HR software, such as SAP SuccessFactors or BambooHR, the model can help HR departments identify employees experiencing high levels of frustration or dissatisfaction. This can lead to proactive interventions to improve workplace satisfaction.

##### 8. Media and Content Recommendation Systems
Use Case: Media platforms (e.g., Netflix or Spotify) can use intensity prediction models to personalize recommendations based on emotional intensity of previously watched or listened-to content.

Integration: The model can be integrated with media platforms to analyze user reviews and feedback, allowing the recommendation engine to suggest content that matches a user's emotional state or preferences based on intensity.

##### 9. Market Research and Consumer Insights
Use Case: Marketing firms could use NLP models to analyze survey responses or focus group discussions for emotional intensity, helping them identify which aspects of a product evoke the strongest reactions.

Integration: Cloud-based intensity prediction models can be built into market research tools that analyze open-ended survey questions. The model could generate insights on consumer feelings about a brand or product, feeding into reports or dashboards.

##### 10. Real-time Feedback for Streaming and Broadcasting
Use Case: Live broadcasters and streamers can monitor audience reactions in real-time, detecting highly intense feedback during important moments (e.g., in sports, gaming, or political broadcasts).

Integration: This can be integrated into social media monitoring tools or live streaming platforms (like Twitch) where feedback is gathered and analyzed in real-time. When the model detects highly intense reactions, broadcasters can adjust content or respond to audience engagement quickly.

##### Integration Strategies in Real-World Systems

##### Cloud Hosting via API Services:

Deploying the model as a microservice via cloud platforms like AWS, Azure, or GCP, accessible via RESTful APIs. This allows different systems to easily interact with the model for real-time or batch predictions.

##### Data Pipelines:

Integrating the model into data processing pipelines, where it becomes part of the end-to-end data workflow. Incoming customer reviews or social media posts are cleaned, preprocessed, passed to the model, and predictions are fed back to user dashboards.

##### Plugging into CRMs:

Customer Relationship Management (CRM) tools like Salesforce or HubSpot can embed this model into their feedback processing modules, enabling automated responses or escalation workflows based on intensity prediction.

##### Mobile and Web Application Integration:

Cloud-deployed models can be integrated into mobile apps or web interfaces via SDKs, enabling businesses to embed real-time intensity analysis directly into their platforms.

##### Streaming Analytics:

Using services like AWS Kinesis or Azure Stream Analytics, the intensity model can process streams of social media data or live customer feedback in real-time, generating dynamic dashboards or sending alerts.
