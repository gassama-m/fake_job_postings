import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Read the data
df = pd.read_excel("C:/Users/Lenovo/Desktop/staj/assign2/preprocessed_data.xlsx")
# Count the number of real (0) and fake (1) job postings
fraud_counts = df['fraudulent'].value_counts()

# Create labels
labels = ['Real (0)', 'Fake (1)']

# Plot pie chart
fraud_counts.plot(kind='pie', labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4682B4', '#B22222'])
plt.title(' Figure 1.1: Proportion of Real vs Fake Job Postings')
plt.ylabel('')  # Hide y-label
plt.savefig('fraud_pie_chart.png')
#plt.show()


# Reshape into long format
df_long = pd.melt(
    df,
    id_vars=['fraudulent'],
    value_vars=['min_salary_capped', 'max_salary_capped'],
    var_name='Salary_Type',
    value_name='Salary'
)

# Optional: rename for prettier labels
df_long['Salary_Type'] = df_long['Salary_Type'].map({
    'min_salary_capped': 'Min Salary',
    'max_salary_capped': 'Max Salary'
})
#rename x axis
df_long['fraudulent'] = df_long['fraudulent'].map({
    0: 'Real',
    1: 'Fake'
})


# Plot
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
plt.savefig('salary.png')
plt.show()


cnt=2
for i in range(11, 19):
    if i == 17:
        continue
    cnt+=1
    col_name = df.columns[i]

    # Crosstab counts per fraudulent group per category
    ct = pd.crosstab(df['fraudulent'], df[col_name], dropna=False)

    # Convert columns to string
    ct.columns = ct.columns.astype(str).fillna('Missing')
    # Special rules:
    if ct.shape[1] == 2:
        # If exactly 2 categories, rename 0 and 1 to No and Yes
        rename_map = {}
        if '0' in ct.columns:
            rename_map['0'] = 'No'
        if '1' in ct.columns:
            rename_map['1'] = 'Yes'
        ct = ct.rename(columns=rename_map)
    elif ct.shape[1] > 2:
        # If more than 2 categories, replace any '0' label with Missing
        if '0' in ct.columns:
            ct = ct.rename(columns={'0': 'Missing'})

    # Compute row sums
    row_sums = ct.sum(axis=1)

    # Compute percentages
    ct_percents = ct.div(row_sums, axis=0) * 100

    # Identify categories where percentage <5% in *both* fraud labels
    small_cats = [
        col for col in ct_percents.columns
        if (ct_percents.loc[:, col] < 5).all()
    ]

    # If any small categories found, relabel them
    if small_cats:
        # Relabel in the original counts
        ct_renamed = ct.rename(columns={cat: 'less than 5%' for cat in small_cats})
        # Re-aggregate
        ct = ct_renamed.T.groupby(level=0).sum().T


    # Recompute percentages after aggregation
    row_sums = ct.sum(axis=1)
    ct_percents = ct.div(row_sums, axis=0) * 100

    # Sort columns alphabetically for consistent colors
    sorted_cols = sorted(ct_percents.columns.astype(str))
    ct_percents = ct_percents.reindex(sorted_cols, axis=1)

    # Map fraudulent values 0/1 to Real/Fake
    x_labels = ct_percents.index.map({0: 'Real', 1: 'Fake'}).tolist()

    # Pick color palette
    n_colors = len(ct_percents.columns)
    palette = sns.color_palette("Paired", n_colors=n_colors)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(ct_percents))

    for j, category in enumerate(ct_percents.columns):
        values = ct_percents[category].values
        ax.bar(
            x_labels,
            values,
            bottom=bottom,
            label=str(category)[:30],
            color=palette[j % len(palette)]
        )
        bottom += values

    if i==18:i=17
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Fraudulent')
    ax.set_title(f'Figure 1.{cnt}: Distribution of {col_name} by fraudulence of postings')
    ax.set_ylim(0, 100)

    ax.legend(title=col_name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{col_name}_by_fraudulent.png')
    plt.show()
