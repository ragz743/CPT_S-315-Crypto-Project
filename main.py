from process_data import process_data

def main():
    # Load Data
    sequence_length = 30
    X, y = process_data(sequence_length)


if __name__ == "__main__":
    main()