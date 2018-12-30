

class EarlyStopping():

    def __init__(self, patience: int = 3):
        """

        :param patience: using training stop decision
        """

        self.best_metrics = None
        self.bad_epochs = 0
        self.patience = patience

    def decision_stop(self, metrics) -> bool:
        """

        :param metrics: value of metrics
        :return: stop: True, don't stop: False
        """

        if self.patience <= 0:
            return False

        if self.best_metrics is None:
            self.best_metrics = metrics
            return False

        if metrics > self.best_metrics:
            self.best_metrics = metrics
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            return True

        return False
