from ccvar import CCVAR
from src.core import BaseModel
import numpy as np


class CCVARPartial(CCVAR):
    pass

class CCVARModel(BaseModel):
    def __init__(self, algorithmParam, cellularComplex):
        ccvarParams = (algorithmParam, cellularComplex)
        algorithm = CCVAR(*ccvarParams)
        initial_params = []
        for key in algorithm._data_keys:
            initial_params.append(algorithm._theta[key])
        initial_params = np.vstack(initial_params)
        super().__init__(initial_params=initial_params, algorithm=algorithm)
        self._param_slices = []
        self._param_length = 0
        self._eta = dict()

    def get_gradient(self, aggregated_data, **kwargs):

        featureDict = self._algorithm._feature_gen()
        grad = []

        inputData = aggregated_data.get_data()

        for key in self._algorithm._data_keys:
            if key not in inputData: continue
            
            S = featureDict[key]
            target = inputData[key].reshape(-1,1)

            self._algorithm._update_state(key, S, target)
            self._eta[key] = self._algorithm._compute_step_size(key)
            curr_grad = self._algorithm._get_gradient(key)

            grad.append(curr_grad)

            try:
                self._param_slices[key]
            except:
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
    
    def update_params(self, update_term):
        super().update_params(update_term=update_term)
        self.set_params(new_params = self._params)


    
    def set_params(self, new_params):
        super().set_params(new_params)
        # import pdb; pdb.set_trace()
        for key in self._algorithm._data_keys:
            param_slice = self._param_slices[key]
            new_param = new_params[param_slice].reshape(-1,1)
            new_param = self._algorithm.soft_threshold(self._eta[key], new_param, self._algorithm._lambda)
            self._algorithm._theta[key] = new_param
        self._params = new_params

        # print(f"New Params: {new_param}")

    def estimate(self, input_data, **kwargs):
        return self._algorithm.forecast(steps = kwargs.get('steps', 1))
        # return self._algorithm._data
        # return self._algorithm._theta




    
    
