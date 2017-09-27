class State:
    
    def __init__(self, mean, covariance, weight):
        self.mean = mean
        self.covariance = covariance
        self.weight = weight