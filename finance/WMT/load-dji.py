import csv
import torch

def load_csv(csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)
    return data

data = load_csv('^dji.csv')

print(data)

# Convert the data to PyTorch tensors
features = torch.tensor(data[0:, :-1])
labels = torch.tensor(data[0:, -1])

# Print the data
print(features)
print(labels)
