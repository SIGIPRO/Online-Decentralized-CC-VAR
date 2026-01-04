from ccvar import CCVAR
from src.core import BaseModel
import numpy as np

class CCVARModel(BaseModel):
    def __init__(self, ccvarParams):
        algorithm = CCVAR(*ccvarParams)
        super().__init__(initial_params=algorithm._theta, algorithm=algorithm)
        self._param_slices = []
        self._param_length = 0

    def get_gradient(self, aggregated_data, **kwargs):

        featureDict = self._algorithm._feature_gen()
        grad = []

        inputData = aggregated_data.get_data()
        ## NOTE: give external_data even in noncommunication rounds as 0, or any of the heuristics.

        for key in self._algorithm._data_keys:
            if key not in inputData: continue

   
            
            S = featureDict[key]
            target = inputData[key].reshape(-1,1)

            self._algorithm._update_state(key, S, target)
            curr_grad = self._algorithm._get_gradient(key)

            grad.append(curr_grad)

            if self._param_slices == []:
                first_index = self._param_length
                last_index = first_index + curr_grad.shape[0]
                self._param_slices.append(slice(first_index, last_index))
                self._param_length += curr_grad.shape[0]

            # NEW (MATLAB Equivalent):
            old_data = self._algorithm._data[key][:, 1:]
            
            # current_step_pred[k] is flat (N,), make it (N, 1)
            new_col = inputData[key].reshape(-1, 1)
            
            self._algorithm._data[key] = np.hstack([old_data, new_col])

        grad = np.vstack(grad).flatten()   
        return grad
    

    
    def set_params(self, new_params):
        super().set_params(new_params)
        for key in self._algorithm._data_keys:
            param_slice = self._algorithm._param_slices[key]
            self._algorithm._theta_dict[key] = new_params[param_slice].reshape(-1,1)

    def estimate(self, input_data, **kwargs):
        return self._algorithm._forecast(steps = kwargs.get('steps', 1))




    
    
