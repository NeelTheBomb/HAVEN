import math
import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience:  Number of epochs to wait if no improvement in validation loss and then stop the training.
        :param min_delta: A minimum decrease in the validation loss to qualify as an improvement, i.e. an decrease of less than or equal to min_delta, will count as no improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.current_best_loss = math.inf  # infinity
        self.current_best_model = None # to store the model_params with lowest validation loss
        self.early_stop = False

    def __call__(self, model, val_loss):
        if self.current_best_loss - val_loss > self.min_delta:
            # validation loss decreased more than the threshold. There is improvement.
            # Reset the current_best_loss to the new val loss
            # Store the model_params as the current_best_model
            # Reset the patience_counter
            self.current_best_loss = val_loss
            self.current_best_model = copy.deepcopy(model)
            self.patience_counter = 0
        else:
            # self.current_best_loss - val_loss is <= self.min_delta
            # No improvement in the validation loss
            # increment the patience_counter
            self.patience_counter += 1
            print(f"Early stopping counter: {self.patience_counter} / {self.patience}")

            if self.patience_counter >= self.patience:
                print("Early STOP: Early stopping threshold reached.")
                self.early_stop = True

    def reset(self):
        self.early_stop = False
        self.patience_counter = 0
        self.current_best_loss = math.inf
        self.current_best_model = None

    def get_current_best_model(self):
        return self.current_best_model