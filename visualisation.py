import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string, os


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("preprocessed_dataset.xlsx")


fraud_counts = df['fraudulent'].value_counts()
labels = ['Real (0)', 'Fake (1)']

fraud_counts.plot(
    kind='pie',
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#4682B4', '#B22222']
)
plt.title('Figure 1.1: Proportion of Real vs Fake Job Postings')
plt.ylabel('')
plt.savefig('fraud_pie_chart.png', bbox_inches='tight')
plt.show()



df_long = pd.melt(
    df,
    id_vars=['fraudulent'],
    value_vars=['min_salary_capped', 'max_salary_capped'],
    var_name='Salary_Type',
    value_name='Salary'
)

df_long['Salary_Type'] = df_long['Salary_Type'].map({
    'min_salary_capped': 'Min Salary',
    'max_salary_capped': 'Max Salary'
})
df_long['fraudulent'] = df_long['fraudulent'].map({
    0: 'Real',
    1: 'Fake'
})

plt.figure(figsize=(8, 6))
sns.violinplot(
    x='fraudulent',
    y='Salary',
    hue='Salary_Type',
    data=df_long,
    split=True,
    inner='quartile',
    palette='Set2'
)
plt.xlabel('Fraudulent Label')
plt.ylabel('Salary')
plt.title('Figure 1.2: Minimum and Maximum Salaries by Fraudulence Type')
plt.legend(title='Salary Type')
plt.tight_layout()
plt.savefig('salary.png', bbox_inches='tight')
plt.show()

categorical_columns = [
    'telecommuting',
    'has_company_logo',
    'has_questions',
    'employment_type',
    'required_experience',
    'required_education',
    'function'
]

cnt = 2
for col_name in categorical_columns:
    cnt += 1
    ct = pd.crosstab(df['fraudulent'], df[col_name], dropna=False)
    ct.columns = ct.columns.astype(str).fillna('Missing')

   
   
    if ct.shape[1] == 2:
        rename_map = {}
        if '0' in ct.columns:
            rename_map['0'] = 'No'
        if '1' in ct.columns:
            rename_map['1'] = 'Yes'
        ct = ct.rename(columns=rename_map)
    elif ct.shape[1] > 2:
        if '0' in ct.columns:
            ct = ct.rename(columns={'0': 'Missing'})

    
    
    row_sums = ct.sum(axis=1)
    ct_percents = ct.div(row_sums, axis=0) * 100

    
    small_cats = [col for col in ct_percents.columns if (ct_percents[col] < 5).all()]
    if small_cats:
        ct_renamed = ct.rename(columns={cat: 'less than 5%' for cat in small_cats})
        ct = ct_renamed.T.groupby(level=0).sum().T

    
    ct_percents = ct.div(ct.sum(axis=1), axis=0) * 100
    sorted_cols = sorted(ct_percents.columns.astype(str))
    ct_percents = ct_percents[sorted_cols]

    
    x_labels = ct_percents.index.map({0: 'Real', 1: 'Fake'}).tolist()
    palette = sns.color_palette("Paired", len(ct_percents.columns))

    
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(ct_percents))
    for j, category in enumerate(ct_percents.columns):
        values = ct_percents[category].values
        ax.bar(x_labels, values, bottom=bottom, label=str(category)[:30], color=palette[j])
        bottom += values

    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Fraudulent')
    ax.set_title(f'Figure 1.{cnt}: Distribution of {col_name} by Fraudulence')
    ax.set_ylim(0, 100)
    ax.legend(title=col_name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f'{col_name}_by_fraudulent.png', bbox_inches='tight')
    plt.show()



text_cols = ['description', 'title', 'requirements', 'company_profile', 'benefits']
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    """Tokenize, lowercase, remove stopwords/punctuation/short words."""
    words = str(text).lower().split()
    words = [w.strip(string.punctuation) for w in words]
    words = [w for w in words if w not in stop_words and w not in punctuations and len(w) > 2]
    return words

def plot_top_words(df, text_cols):
    for col in text_cols:
        if col not in df.columns:
            print(f" Column '{col}' not found, skipping.")
            continue

        print(f"\n Generating Top 10 Words comparison for: {col}")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(wspace=0.5)

        for i, label in enumerate([0, 1]):
            subset = df[df['fraudulent'] == label][col].dropna().astype(str)
            all_words = []
            for text in subset:
                all_words.extend(clean_text(text))

            label_name = "Non-Fraudulent" if label == 0 else "Fraudulent"
            color = 'green' if label == 0 else 'red'

            common_words = Counter(all_words).most_common(10)
            if not common_words:
                axes[i].text(0.5, 0.5, f"No words in {label_name}", ha='center', va='center', fontsize=14)
                axes[i].axis('off')
                continue

            words_df = pd.DataFrame(common_words, columns=['word', 'count'])
            sns.barplot(data=words_df, x='count', y='word', ax=axes[i], palette=[color])
            axes[i].set_title(label_name, fontsize=14, fontweight='bold', pad=12)
            axes[i].set_xlabel("Count")
            axes[i].set_ylabel("Word")

            
            for spine in axes[i].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        fig.suptitle(f"Top 10 Most Common Words for '{col}'", fontsize=18, fontweight='bold', y=0.98)
        save_path = os.path.join(output_dir, f"{col}_top_words_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        print(f"Saved top words comparison: {save_path}")


plot_top_words(df, text_cols)
