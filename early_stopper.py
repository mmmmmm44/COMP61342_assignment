class EarlyStopper:
    """A handy early stopper to stop the training process when the validation loss
    does not improve for a certain number of epochs (patience)."""

    def __init__(self, patience=5, delta=0, minimize=True):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            minimize (bool): Whether to minimize or maximize the monitored quantity.
        """
        self.patience = patience
        self.delta = delta
        self.minimize = minimize    # define if the loss (quantity to be monitored) is minimized or maximized
        
        self.counter = 0
        self.best_loss = None       # the quantity to be monitored
        
        self.best_epoch = -1
        self.best_model = None

    def step(self, loss, curr_epoch, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.counter = 0
            self.best_epoch = curr_epoch
            self.best_model = model

        # check if the loss improved
        elif self.minimize and loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            self.best_epoch = curr_epoch
            self.best_model = model
        elif not self.minimize and loss > self.best_loss + self.delta:
            self.best_loss = loss
            self.counter = 0
            self.best_epoch = curr_epoch
            self.best_model = model
        else:
            self.counter += 1

        return self.counter >= self.patience
    
    def get_best_model(self):
        return self.best_model