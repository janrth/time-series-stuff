import numpy as np

class ForecastError(object):
    
    def __init__(self):
        pass

    def ensure_numpy_array(self, y):
        if not isinstance(y, np.ndarray):
            return np.array(y)
        return y

    # Mean Absolute Percentage Error (MAPE)
    def mape(self, y_true, y_pred): 
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return np.mean(np.abs((y_true - y_pred)) / y_true)

    # Mean Absolute Error (MAE)
    def mae(self, y_true, y_pred):
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    # Symmetric Mean Absolute Percentage Error (sMAPE)
    def smape(self, y_true, y_pred):
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return np.mean(2.0 * np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) + np.finfo(float).eps)) 
    
    def forecast_bias(self, y_true, y_pred):
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return (np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true)
    
    def mase(self, y_true, y_pred, y_naive):
        y_true, y_pred, y_naive = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred), self.ensure_numpy_array(y_naive)
        return np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true - y_naive))
    
    # Mean Squared Error (MSE)
    def mse(self, y_true, y_pred):
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    def rmse(self, y_true, y_pred):
        y_true, y_pred = self.ensure_numpy_array(y_true), self.ensure_numpy_array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))



