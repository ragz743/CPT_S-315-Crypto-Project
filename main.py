from evaluate_predictions import evaluate_predictions
from plot_predictions import plot_predictions
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
    # Train the model and get predictions and actual targets
    predicted_original, y_tensor = tensor(processed_data_object, sequence_x, sequence_y)

    # compute the target in original scale
    close_mean = processed_data_object["close_mean"]
    close_std = processed_data_object["close_std"]
    actual_original = y_tensor * close_std + close_mean

    # evaluate predictions
    mae, rmse, mape = evaluate_predictions(actual_original.detach().numpy(),
                                       predicted_original.detach().numpy())
    print("Mean Absolute Percentage Error:", mape)

    # Plot the predictions vs. actual prices
    x_axis = range(len(actual_original.detach().numpy()))
    plot_predictions(x_axis, actual_original.detach().numpy(), predicted_original.detach().numpy())


if __name__ == "__main__":
    main()
