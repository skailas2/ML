import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of GloVe vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################

def load_tsv_dataset(file):
    # Load the dataset using numpy, labels are the first column, reviews are the second column
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8', dtype='l,O')
    labels = []
    words = []

    # Separate the labels and words (reviews)
    for i in dataset:
        i = list(i)
        labels.append(i[0])
        words.append(i[1:])
    
    return labels, words

def load_feature_dictionary(file):
    word_to_vector = {}
    with open(file) as f:
        reader = csv.reader(f, delimiter='\t')  # Specify tab as the delimiter
        for row in reader:
            word = row[0]  # First column is the word
            vector = np.array(row[1:], dtype=float)  # The rest is the vector
            word_to_vector[word] = vector  # Add to dictionary
    return word_to_vector

def get_average_vectors(reviews, word_to_vector):
    output = {}
    for i, word_list in enumerate(reviews):
        sum_vector = np.zeros(VECTOR_LEN)  # Initialize a vector of size 300 for summing
        total_count = 0

        for w in word_list:
            strings = w.split()
            filtered = [s for s in strings if s in word_to_vector]  # Use only words in the GloVe embeddings
            for word in filtered:
                sum_vector += word_to_vector[word]  # Add the word's embedding
                total_count += 1  # Increment the word count

        # Only normalize if the sentence has valid words
        if total_count > 0:
            output[i] = sum_vector / total_count  # Normalize by the number of words in the sentence
        else:
            output[i] = np.zeros(VECTOR_LEN)  # Return a zero vector if no valid words found
    return output

def write_output(file_path, labels, output):
    with open(file_path, mode='w', newline='') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        for label, sum_vector in zip(labels, output.values()):
            # Round the summed vector to 6 decimal places
            sum_vector = np.round(sum_vector, 6)
            # Create the row: label followed by the feature values
            row = [label] + list(sum_vector)
            # Write the row to the CSV file
            writer.writerow(row)

# Add debug prints in the main section
if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, help='path to output .tsv file for training data feature extractions')
    parser.add_argument("validation_out", type=str, help='path to output .tsv file for validation data feature extractions')
    parser.add_argument("test_out", type=str, help='path to output .tsv file for test data feature extractions')
    args = parser.parse_args()

    # Load datasets
    train_labels, train_reviews = load_tsv_dataset(args.train_input)
    validation_labels, validation_reviews = load_tsv_dataset(args.validation_input)
    test_labels, test_reviews = load_tsv_dataset(args.test_input)

    # Load GloVe embeddings
    word_to_vector = load_feature_dictionary(args.feature_dictionary_in)

    # Compute average vectors
    train_vectors = get_average_vectors(train_reviews, word_to_vector)
    validation_vectors = get_average_vectors(validation_reviews, word_to_vector)
    test_vectors = get_average_vectors(test_reviews, word_to_vector)

    # Write outputs to the corresponding files
    write_output(args.train_out, train_labels, train_vectors)
    write_output(args.validation_out, validation_labels, validation_vectors)
    write_output(args.test_out, test_labels, test_vectors)

    print(f"Outputs successfully written to {args.train_out}, {args.validation_out}, and {args.test_out}.")
