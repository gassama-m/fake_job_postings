import pandas as pd

path="C:/Users/Lenovo/Desktop/staj/fake_job_postings.csv"
df = pd.read_csv(path)

print(df.shape)

#Replacing all the missing values by '0-0' in salary_range column
df['salary_range'] = df['salary_range'].fillna('0-0')


#creating min_salary and max_salary columns right after salary_range
salary_range_indx=df.columns.get_loc('salary_range')
df.insert(salary_range_indx+1, 'min_salary', 0)
df.insert(salary_range_indx+2, 'max_salary', 0)

# Spliting on hyphen or en dash
df[['min_salary', 'max_salary']] = df['salary_range'].str.split(r'[-–]', expand=True)

# Converting to numeric 
df['min_salary'] = pd.to_numeric(df['min_salary'], errors='coerce')
df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')

#printing the values for table 1
print(df.info())
print(df.describe(include='all'))

print("\nPercentage of missing values:")
print(df.isna().mean() * 100)

#dropping the dduplications
df.drop_duplicates(inplace=True)
#replacing all the missing values by 0
df.fillna(0, inplace=True)
df['required_education'] = df['required_education'].replace("Unspecified", 0)
#outlier handling using IQR method for min_salary and max_salary columns
arr = ['min', 'max']

for i in arr:
    col = i + '_salary'
    
    #considering only non-zero values for IQR calculation
    non_zero_values = df.loc[df[col] != 0, col]
    
    Q1 = non_zero_values.quantile(0.25)
    Q3 = non_zero_values.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    print(lower_bound, upper_bound)
    
    #outliers detection ignoring zeros
    outliers = df[(df[col] != 0) & ((df[col] < lower_bound) | (df[col] > upper_bound))]
    
    print(f"Outliers detected in '{col}':")
    print(outliers[[col]])
    
    #outliers capping
    df[i + '_salary_capped'] = df[col].clip(lower=lower_bound, upper=upper_bound)

#Exporting the output to an excel file
#df.to_excel("preprocessed_data.xlsx", index=False)