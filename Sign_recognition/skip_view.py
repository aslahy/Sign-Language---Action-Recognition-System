import matplotlib.pyplot as plt
from collections import Counter

# Read the skipped images text file
with open('skipped_images.txt', 'r') as f:
    skipped_images = f.readlines()

# Extract the lowercase letters from the skipped image IDs
labels = [line.split('/')[0].lower() for line in skipped_images]

# Count the occurrences of each letter
label_counts = Counter(labels)

# Sort the counts by alphabetical order
sorted_labels = sorted(label_counts.keys())
sorted_counts = [label_counts[label] for label in sorted_labels]

# Plot the data
plt.figure(figsize=(10, 6))
plt.bar(sorted_labels, sorted_counts, color='skyblue')
plt.xlabel('Lowercase Letters', fontsize=14)
plt.ylabel('Number of Skipped Images', fontsize=14)
plt.title('Skipped Images by Lowercase Letter', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
