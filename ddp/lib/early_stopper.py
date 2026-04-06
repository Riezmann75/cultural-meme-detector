class EarlyStopper:
    def __init__(self, patience=10, delta=0.001):
        """
        :param patience: Number of epochs to wait after last improvement.
        :param delta: Minimum change in the monitored metric to qualify as an improvement.
        :param path: Filepath to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
