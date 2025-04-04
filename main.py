from process_data import process_data
from sequence import sequence_data
from training import tensor


def main():
    # process data first
    processed_data_object = process_data()
    print("Processed Data Results:")
    print(processed_data_object["df"])

    # sequence
    sequence_length = 30
    sequence_x, sequence_y = sequence_data(sequence_length, processed_data_object["df"])
    '''
    print("Sequence Data Results:")
    print("Sequence Data X:")
    print(X)
    print("Sequence Data Y:")
    print(y)
    '''
    tensor(processed_data_object, sequence_x, sequence_y)

if __name__ == "__main__":
    main()
