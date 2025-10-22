import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load dataset
path = "dataforcormatrx.xlsx"
df = pd.read_excel(path)

# Select numeric columns and exclude specific ones
numeric_cols = (
    df.select_dtypes(include='number')
      .drop(columns=['job_id', 'min_salary', 'max_salary'], errors='ignore')
)

# Compute correlation matrix
corr = numeric_cols.corr()

# Reorder columns to have 'fraudulent' last if it exists
if 'fraudulent' in corr.columns:
    cols = [col for col in corr.columns if col != 'fraudulent'] + ['fraudulent']
    corr_reordered = corr.loc[cols, cols]
else:
    corr_reordered = corr

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_reordered,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75}
)
#plt.title("Correlation Matrix", fontsize=14)
plt.tight_layout()
# plt.savefig("Correlation_matrix.png", dpi=300)
plt.show()