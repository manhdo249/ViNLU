import csv


# Function to split the CSV file
def split_csv(ground_truth, label, seq):
    with open(ground_truth, 'r') as infile:
        reader = csv.reader(infile)
        with open(label, 'w', newline='') as f1, open(seq, 'w', newline='') as f2:
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)

            for row in reader:
                if len(row) == 2:  # Ensure there are two columns
                    writer1.writerow([row[0]])  # Write first part to label
                    writer2.writerow([row[1].strip()])  # Write second part to seq


# File paths
ground_truth = 'ground_truth.csv'
label = 'label'
seq = 'seq.out'

# Split the file
split_csv(ground_truth, label, seq)
