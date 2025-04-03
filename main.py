from process_data import process_data
from sequence import sequence_data


def main():
    # process data first
    processed_data = process_data()
    print("Processed Data Results:")
    print(processed_data)

    # sequence
    sequence_length = 30
    X, y = sequence_data(sequence_length, processed_data)
    print("Sequence Data Results:")
    print("Sequence Data X:")
    print(X)
    print("Sequence Data Y:")
    print(y)

if __name__ == "__main__":
    main()
