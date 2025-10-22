import pandas as pd
import numpy as np
import re, string
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


path = "/content/fake_job_postings.csv"
df = pd.read_csv(path)
print("Original dataset shape:", df.shape)

# misssing salary
df['salary_range'] = df['salary_range'].fillna('0-0')

#  new columns min/max salary
salary_range_idx = df.columns.get_loc('salary_range')
df.insert(salary_range_idx + 1, 'min_salary', 0)
df.insert(salary_range_idx + 2, 'max_salary', 0)

df[['min_salary', 'max_salary']] = df['salary_range'].str.split(r'[-–]', expand=True)
df['min_salary'] = pd.to_numeric(df['min_salary'], errors='coerce')
df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')

# drop duplicate and replace missing values
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)
df['required_education'] = df['required_education'].replace("Unspecified", 0)

# outliers handling
for i in ['min', 'max']:
    col = i + '_salary'
    non_zero_values = df.loc[df[col] != 0, col]
    Q1 = non_zero_values.quantile(0.25)
    Q3 = non_zero_values.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    df[i + '_salary_capped'] = df[col].clip(lower=lower_bound, upper=upper_bound)

# drop unecessary columns
useless_columns = ['job_id',  'salary_range', 'location', 'min_salary', 'max_salary' ]
useless_columns = [col for col in useless_columns if col in df.columns]
df.drop(columns=useless_columns, inplace=True)
print(f"Dropped useless columns: {useless_columns}")

# clean text
text_columns = ['title', 'description', 'requirements', 'company_profile', 'benefits']
stopwords = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in stopwords]
    return " ".join(tokens)

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(clean_text).fillna("")

# Add text length features
df['desc_length'] = df['description'].apply(len)
df['req_length'] = df['requirements'].apply(len)
df['bene_length'] = df['benefits'].apply(len)

# drop rare categories (frequence<5%)
non_text_columns = [c for c in df.columns if c not in text_columns]
min_freq = 0.05
for col in non_text_columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        counts = df[col].value_counts(normalize=True)
        rare_categories = counts[counts < min_freq].index
        df[col] = df[col].replace(rare_categories, 'Other')

# tf-idf vectorization
tfidf_title = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')
tfidf_desc  = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
tfidf_req   = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
tfidf_profile = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
tfidf_bene = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')

X_title = tfidf_title.fit_transform(df['title'])
X_desc  = tfidf_desc.fit_transform(df['description'])
X_req   = tfidf_req.fit_transform(df['requirements'])
X_prof  = tfidf_profile.fit_transform(df['company_profile'])
X_bene  = tfidf_bene.fit_transform(df['benefits'])

X_text = hstack([X_title, X_desc, X_req, X_prof])

# structured features
categorical_features = ['employment_type', 'required_experience', 'required_education',  'function', 'department',  'industry']
numeric_features = [ 'min_salary_capped', 'max_salary_capped', 'desc_length', 'req_length']

for col in categorical_features:
    df[col] = df[col].replace(0, "Unknown").fillna("Unknown")
for col in numeric_features:
    df[col] = df[col].fillna(0)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

X_struct = preprocessor.fit_transform(df)

# combine all features
X_final = hstack([X_text, csr_matrix(X_struct)])
X_final = csr_matrix(X_final)

print("✅ Preprocessing complete!")
print("Final feature matrix shape:", X_final.shape)

# save 
df.to_excel("preprocessed_datasonnn.xlsx", index=False)
print("Cleaned data exported → preprocessed_datasonnn.xlsx")
