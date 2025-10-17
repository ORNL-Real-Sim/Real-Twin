import pandas as pd

# Load the CSV file (modify 'your_file.csv' to your file's path)
df = pd.read_csv('inflow.csv')

# Split the first column into separate columns (assuming the column is named 'Numbers')
# Change 'Numbers' to the actual name of your first column if different
expanded_columns = df['solution'].str.split(',', expand=True)

# Rename the new columns (optional)
expanded_columns.columns = [f'col{i+1}' for i in range(expanded_columns.shape[1])]

# Drop the original first column and concatenate the new split columns with the rest of the DataFrame
result_df = pd.concat([expanded_columns, df.drop(columns=['solution'])], axis=1)

# Save the result to a new CSV file
result_df.to_csv('final_inflow.csv', index=False)

print("The CSV file has been updated and saved as 'expanded_file.csv'.")