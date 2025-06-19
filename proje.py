import numpy as numpy
import pandas as pd

path="C:/Users/Lenovo/Desktop/staj/job_postings.csv"
df1=pd.read_csv(path)
column_name="fraudulent"
#create two csv files; one for false job posting and one for real job postions
job_authenticity=["real", "false"]
for i in job_authenticity:
    fraud = df1[df1["fraudulent"] == job_authenticity.index(i)]
    job_path = "C:/Users/Lenovo/Desktop/staj/"+i+"_postings.csv"
    fraud.to_csv(job_path, index=False)

    #Create for each fraudulence a excel file with the number of column as sheet, in every sheet we will have 
    #the count of every entry and their pourcentage
    df2 = pd.read_csv("C:/Users/Lenovo/Desktop/staj/"+i+"_postings.csv")

    with pd.ExcelWriter(i+"byColumn.xlsx") as writer:
         for column in df2.columns:
            nan_count = df2[column].isna().sum()
            nan_percentage = (nan_count / len(df2)) * 100
            nan_row = pd.DataFrame({
                "Count": [nan_count],
                "Percentage (%)": [round(nan_percentage, 2)]
            }, index=["NaN"])

            if column == "job_id":
                counts = df2[column].value_counts(dropna=True)
                id_rows = pd.DataFrame({"Count": counts})
                nan_row = pd.DataFrame({"Count": [nan_count]}, index=["NaN"])
                summary = pd.concat([nan_row, id_rows])
            else:
                col_data = df2[column].dropna()
                counts = col_data.value_counts()
                percentages = (counts / len(col_data)) * 100
                value_rows = pd.DataFrame({
                    "Count": counts,
                    "Percentage (%)": percentages.round(2)
                })
                summary = pd.concat([nan_row, value_rows])

            summary.to_excel(writer, sheet_name=str(column)[:31])
print("okkk)")