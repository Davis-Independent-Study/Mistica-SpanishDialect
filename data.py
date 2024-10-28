import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data/Combined.tsv', sep='\t')

# List of countries to split data for
countries = ["AR", "CL", "CO", "ES", "MX", "US", "UY", "VE"]

# Split ratios
train_ratio, test_ratio, val_ratio = 0.8, 0.1, 0.1

# Iterate over each country to split data and add the label column
split_data = {country: {} for country in countries}

for country in countries:
    # Filter data for the specific country and add label column
    country_data = data[data['country'] == country].copy()
    country_data['label'] = 1  # Label as 1 for specific country
    
    # Shuffle data and split into train, test, and validation ensuring non-overlapping splits
    country_data = country_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_end = int(train_ratio * len(country_data))
    test_end = train_end + int(test_ratio * len(country_data))

    train_data = country_data.iloc[:train_end]
    test_data = country_data.iloc[train_end:test_end]
    valid_data = country_data.iloc[test_end:]

    # Add label 0 for the other countries and sample accordingly for test and validation sets
    other_data = data[data['country'] != country].copy()
    other_data['label'] = 0

    # Combine non-overlapping data for each split
    train_data = pd.concat([train_data, other_data.sample(frac=(1 - train_ratio), random_state=42)])
    test_data = pd.concat([test_data, other_data.sample(frac=(1 - test_ratio), random_state=42)])
    valid_data = pd.concat([valid_data, other_data.sample(frac=(1 - val_ratio), random_state=42)])

    # Store the splits
    split_data[country]['train'] = train_data
    split_data[country]['test'] = test_data
    split_data[country]['valid'] = valid_data

# Save the split datasets as TSV files
for country in split_data:
    for split_type in ['train', 'test', 'valid']:
        file_path = f'{country}_{split_type}_no_overlap.tsv'
        split_data[country][split_type].to_csv(file_path, index=False, sep='\t')

print("Data splits have been successfully saved as TSV files.")


