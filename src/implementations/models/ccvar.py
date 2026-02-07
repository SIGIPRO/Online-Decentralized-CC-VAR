from ccvar import CCVAR
from src.core import BaseModel
import numpy as np


class CCVARPartial(CCVAR):
    def __init__(self, algorithmParam, cellularComplex, theta_initializer = None):
        self.__algorithm_parameter_setup(algorithmParam=algorithmParam)
        self._data_keys = sorted([i for i, en in enumerate(self._data_enabler) if en])

        self._phi = dict()
        self._r = dict()
        self._theta = dict()
        
        # 1. Generic Topology Construction
        self._construct_laplacian(cellularComplex)

        # 2. Data Buffers
        self._data_initializer()

        # 3. Weights
        if theta_initializer is None:
            self._theta_initializer = self.__zero_initializer
        else:
            self._theta_initializer = theta_initializer

        self._allocate_state_variables()

    def update(self, inputData):
        featureDict = self._feature_gen()

         # import pdb; pdb.set_trace()
        for key in self._data_keys:
            if key not in inputData: continue
            
            S = featureDict[key]
            # Ensure target is (N, 1)
            target = inputData[key].reshape(-1, 1)
            target = target[self._out_idx[key],:]
            
            # --- Optimization Hooks ---
            self._update_state(key, S, target)
            eta = self._compute_step_size(key)
            self._apply_descent_step(key, eta)

             # NEW (MATLAB Equivalent):
            old_data = self._data[key][:, 1:]
            
            # current_step_pred[k] is flat (N,), make it (N, 1)
            new_col = inputData[key].reshape(-1, 1)
            
            self._data[key] = np.hstack([old_data, new_col])
    def forecast(self, steps = 1):
        """
        1-step forecasting. Definition of multi-step forecasting in this case a little bit problematic.
        """

        preds = {k: np.zeros((self._N[k], steps)) for k in self._data_keys}
        feats = self._feature_gen()
        for k in self._data_keys:
            if self._theta[k] is None: continue
            preds[k] = (feats[k] @ self._theta[k]).flatten()

        return preds
            

    def _data_initializer(self):
        self._data = dict()
        self._norm_scale = dict()
        self._bias = dict()
        for key in self._data_keys:
            self._data[key] = np.zeros(shape=(self._Nin[key], self._P))
            self._norm_scale[key] = 0
            if self._bias_enabler:
                self._bias[key] = np.ones(shape=(self._Nout[key], 1))
            else:
                self._bias[key] = np.empty(shape=(self._Nout[key], 0))

    def _construct_laplacian(self, cellularComplex):
        self._Nin = dict()
        self._Nout = dict()
        self._features = dict()
        self._Rk = dict()


        for k in self._data_keys:
            self._features[k] = {}
            feature_dim_accum = 0

            self._Nin[k] = len(self._in_idx[k])
            self._Nout[k] = len(self._out_idx[k])



            B_down = cellularComplex.get(k)
            B_down = B_down[self._in_idx.get(k - 1), self._in_idx.get(k)]
            B_up = cellularComplex.get(k + 1)
            B_up = B_up[self._in_idx.get(k), self._in_idx.get(k + 1)]

            L_lower = None
            L_upper = None
            
            if B_down is not None:
                L_lower = B_down.T @ B_down
            if B_up is not None:
                L_upper = B_up @ B_up.T

            mu_val = self._mu[k] if k < len(self._mu) else 0
            self._Rk[k] = np.eye(self._Nout[k])

                # Add regularization for L_lower/L_upper
            if L_lower is not None:
                 mu_l = mu_val[0] if isinstance(mu_val, (list, tuple)) else mu_val
                 self._Rk[k] += mu_l * L_lower[self._out_idx.get(k), self._out_idx.get(k)]
            if L_upper is not None:
                 mu_u = mu_val[1] if isinstance(mu_val, (list, tuple)) else mu_val
                 self._Rk[k] += mu_u * L_upper[self._out_idx.get(k), self._out_idx.get(k)]

            # --- Feature Dimension accumulation matches Order in __generic_features ---
            # 1. Lower Neighbor
            if L_lower is not None:
                K_val = self._K[k] if k < len(self._K) else 2
                K_l = K_val[0] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["sl"] = self.__ll_gen(L_lower, K_l)
                
                # Check for Lower Neighbor existence
                if (k-1) in self._data_keys:
                    self._features[k]["l"] = self.__multiply_matrices_blockwise(self._features[k]["sl"], B_down.T)
                    self._features[k]["l"] = self._features[k]["l"][self._out_idx.get(k),:,:]
                    feature_dim_accum += K_l * self._P

                self._features[k]["sl"] = self._features[k]["sl"][self._out_idx.get(k),:,:]
                
            
            # 2. Self (Lower)
            if L_lower is not None:
                 feature_dim_accum += K_l * self._P 

            # 3. Self (Upper)
            if L_upper is not None:
                K_val = self._K[k] if k < len(self._K) else 2
                K_u = K_val[1] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["su"] = self.__ll_gen(L_upper, K_u)
                feature_dim_accum += K_u * self._P
            
            # 4. Upper Neighbor
            if L_upper is not None:
                if (k+1) in self._data_keys:
                    self._features[k]["u"] = self.__multiply_matrices_blockwise(self._features[k]["su"], B_up)
                    self._features[k]["u"] = self._features[k]["u"][self._out_idx.get(k),:,:]
                    feature_dim_accum += K_u * self._P

                self._features[k]["su"] = self._features[k]["su"][self._out_idx.get(k),:,:]
                

            # 5. Bias
            feature_dim_accum += int(self._bias_enabler)

            self._features[k]["S_dim"] = feature_dim_accum




    def __algorithm_parameter_setup(self, algorithmParam):
        super().__algorithm_parameter_setup(algorithmParam)
        self._in_idx = algorithmParam['in_idx']
        self._out_idx = algorithmParam['out_idx']
        


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




    
    
