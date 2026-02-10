from ccvar import CCVAR
from src.core import BaseModel
import numpy as np
import copy


class CCVARPartial(CCVAR):
    def __init__(self, algorithmParam, cellularComplex, theta_initializer = None):
        self._algorithm_parameter_setup(algorithmParam=algorithmParam)
        self._data_keys = sorted([i for i, en in enumerate(self._data_enabler) if en])

        self._phi = dict()
        self._r = dict()
        self._theta = dict()
        # import pdb; pdb.set_trace()
        
        # 1. Generic Topology Construction
        self._construct_laplacian(cellularComplex)

        # 2. Data Buffers
        self._data_initializer()

        # 3. Weights
        if theta_initializer is None:
            self._theta_initializer = self._CCVAR__zero_initializer
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
            
            # Keep internal buffer aligned with the model's input-indexed local view.
            new_col = inputData[key].reshape(-1, 1)
            if new_col.shape[0] != old_data.shape[0]:
                in_idx = self._in_idx.get(key, [])
                if in_idx:
                    new_col = new_col[np.asarray(in_idx, dtype=int), :]
            
            self._data[key] = np.hstack([old_data, new_col])
    def _in_in_generic_features(self, key, x_self, x_lower, x_upper):
        """
        Constructs S = [Neighbor_Lower, Self_Lower, Self_Upper, Neighbor_Upper, Bias]
        Order matched to MATLAB: [Lower, Self, Upper, Bias]
        """
        components = []
        
        # 1. Lower Neighbor Features (L * B_k^T * x_{k-1})
        if x_lower is not None and "l" in self._features[key]:
            components.append(self._CCVAR__matrix_vector_bw(self._in_in_features[key]["l"], x_lower))

        # 2. Self Features (Lower Coupling)
        if "sl" in self._features[key]:
             components.append(self._CCVAR__matrix_vector_bw(self._in_in_features[key]["sl"], x_self))
        
        # 3. Self Features (Upper Coupling)
        if "su" in self._features[key]:
             components.append(self._CCVAR__matrix_vector_bw(self._in_in_features[key]["su"], x_self))
        
        # 4. Upper Neighbor Features (L * B_{k+1} * x_{k+1})
        if x_upper is not None and "u" in self._features[key]:
            components.append(self._CCVAR__matrix_vector_bw(self._in_in_features[key]["u"], x_upper))
            
        # 5. Bias
        if self._bias_enabler:
            # In in-in recursion, rows are Nin (not Nout), so bias must match Nin.
            components.append(np.ones((x_self.shape[0], 1)))
        else:
            components.append(np.empty((x_self.shape[0], 0)))
        
        return np.hstack(components)
    def _in_in_feature_gen(self):
        """
        Generic Feature Generation.
        CRITICAL FIX: Normalization is Column-Wise (axis=0).
        """
        featureDict = dict()

        for key in self._data_keys:
            x_self = self._data[key]
            x_lower = self._data.get(key - 1) if (key - 1) in self._data else None
            x_upper = self._data.get(key + 1) if (key + 1) in self._data else None
            
            # Generate Raw Features
            S = self._in_in_generic_features(key, x_self, x_lower, x_upper)

            # Normalization
            if self._FeatureNormalzn:
                # 1. Compute Squared Norm per Column (Vector of size FeatDim)
                # CRITICAL: axis=0 prevents scalar summation of the whole matrix
                S_n = np.sum(S**2, axis=0)
                
                # 2. Floor to prevent division by zero (Matches MATLAB 0.001)
                S_n[S_n == 0] = 0.001
                
                # 3. Compute Signal Variance (Scalar) of the most recent lag
                varV = np.sum(x_self[:, -1]**2)
                
                # 4. Update Running Scale
                self._norm_scale[key] = (1 - self._b) * self._norm_scale[key] + self._b * np.sqrt(varV)
                
                # 5. Broadcast Division and Scale
                # (N, F) / (F,) -> Each column divided by its norm
                featureDict[key] = (S / np.sqrt(S_n)) * self._norm_scale[key]
            else:
                featureDict[key] = S

        return featureDict

    def forecast(self, steps = 1):
        """
        1-step forecasting. Definition of multi-step forecasting in this case a little bit problematic.
        """
    
        const_obj = copy.deepcopy(self)
        preds = {k: np.zeros((const_obj._Nout[k], steps)) for k in self._data_keys}
        
        for s in range(steps):
            if s == steps - 1:
                feats = const_obj._feature_gen()
            else:
                feats = const_obj._in_in_feature_gen()

            for k in self._data_keys:
                if self._theta[k] is None: continue
                curr_preds = (feats[k] @ const_obj._theta[k]).flatten()
                if s == steps - 1:
                    preds[k][:,s] = curr_preds
                else:
                    preds[k][:,s] = curr_preds[const_obj._out_idx[k]]

                    old_data = const_obj._data[k][:, 1:]
                    new_col = curr_preds.reshape(-1,1)
                    const_obj._data[k] = np.hstack([old_data, new_col])

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
        self._in_in_features = dict()
        self._Rk = dict()


        for k in self._data_keys:
            self._features[k] = {}
            self._in_in_features[k] = {}
            feature_dim_accum = 0

            self._Nin[k] = len(self._in_idx[k])
            self._Nout[k] = len(self._out_idx[k])



            B_down = cellularComplex.get(k)
            if B_down is not None:
                B_down = B_down[np.ix_(self._in_idx.get(k - 1), self._in_idx.get(k))]

            B_up = cellularComplex.get(k + 1)
            if B_up is not None:
                B_up = B_up[np.ix_(self._in_idx.get(k), self._in_idx.get(k + 1))]
            

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
                 self._Rk[k] += mu_l * L_lower[np.ix_(self._out_idx.get(k), self._out_idx.get(k))]
            if L_upper is not None:
                 mu_u = mu_val[1] if isinstance(mu_val, (list, tuple)) else mu_val
                 self._Rk[k] += mu_u * L_upper[np.ix_(self._out_idx.get(k), self._out_idx.get(k))]

            # --- Feature Dimension accumulation matches Order in __generic_features ---
            # 1. Lower Neighbor
            if L_lower is not None:
                K_val = self._K[k] if k < len(self._K) else 2
                K_l = K_val[0] if isinstance(K_val, (list, tuple)) else K_val
                
                self._features[k]["sl"] = self._CCVAR__ll_gen(L_lower, K_l)
                self._in_in_features[k]["sl"] = copy.deepcopy(self._features[k]["sl"])
                
                # Check for Lower Neighbor existence
                if (k-1) in self._data_keys:
                    self._features[k]["l"] = self._CCVAR__multiply_matrices_blockwise(self._features[k]["sl"], B_down.T)
                    self._in_in_features[k]["l"] = copy.deepcopy(self._features[k]["l"])
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
                
                self._features[k]["su"] = self._CCVAR__ll_gen(L_upper, K_u)
                self._in_in_features[k]["su"] = copy.deepcopy(self._features[k]["su"])
                feature_dim_accum += K_u * self._P
            
            # 4. Upper Neighbor
            if L_upper is not None:
                if (k+1) in self._data_keys:
                    self._features[k]["u"] = self._CCVAR__multiply_matrices_blockwise(self._features[k]["su"], B_up)
                    self._in_in_features[k]["u"] = copy.deepcopy(self._features[k]["u"])
                    self._features[k]["u"] = self._features[k]["u"][self._out_idx.get(k),:,:]
                    feature_dim_accum += K_u * self._P

                self._features[k]["su"] = self._features[k]["su"][self._out_idx.get(k),:,:]
                

            # 5. Bias
            feature_dim_accum += int(self._bias_enabler)

            self._features[k]["S_dim"] = feature_dim_accum
            self._in_in_features[k]["S_dim"] = feature_dim_accum




    def _algorithm_parameter_setup(self, algorithmParam):
        self._Tstep = algorithmParam.get('Tstep', 1)
        self._mu = algorithmParam.get('mu', [0, (0,0), 0]) 
        self._lambda = algorithmParam.get('lambda', 0.01)
        self._LassoEn = algorithmParam.get('LassoEn', 0)
        self._FeatureNormalzn = algorithmParam.get('FeatureNormalzn', True)
        self._bias_enabler = algorithmParam.get('BiasEn', True)
        self._b = algorithmParam.get('b', 1)
        self._gamma = algorithmParam.get('gamma', 0.98)
        self._P = algorithmParam.get('P', 2)
        self._K = algorithmParam.get('K', [2, (2,2), 2]) 
        self._data_enabler = algorithmParam.get('enabler', [True, True, True])
        # super(CCVARPartial, self).__algorithm_parameter_setup(algorithmParam=algorithmParam)
        self._in_idx = algorithmParam['in_idx']
        self._out_idx = algorithmParam['out_idx']
        

class CCVARPartialIn(CCVARPartial):
    def forecast(self, steps=1):
        """
        Forecast on the full in_idx state for each enabled dimension.
        Returns predictions with shape (Nin, steps), without projecting to out_idx.
        """
        const_obj = copy.deepcopy(self)
        preds = {k: np.zeros((const_obj._Nin[k], steps)) for k in self._data_keys}

        for s in range(steps):
            feats = const_obj._in_in_feature_gen()

            for k in self._data_keys:
                if const_obj._theta[k] is None:
                    continue

                curr_preds = (feats[k] @ const_obj._theta[k]).flatten()
                preds[k][:, s] = curr_preds

                if s < steps - 1:
                    old_data = const_obj._data[k][:, 1:]
                    new_col = curr_preds.reshape(-1, 1)
                    const_obj._data[k] = np.hstack([old_data, new_col])

        return preds


class CCVARPartialModel(BaseModel):
    def __init__(self, algorithmParam, cellularComplex):
        ccvarParams = (algorithmParam, cellularComplex)
        algorithm = CCVARPartial(*ccvarParams)
        initial_params = []
        for key in algorithm._data_keys:
            initial_params.append(algorithm._theta[key])
        initial_params = np.vstack(initial_params)

        super().__init__(initial_params=initial_params, algorithm=algorithm)
        self._param_slices = dict()
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
            target = target[self._algorithm._out_idx[key], :]

            self._algorithm._update_state(key, S, target)
            self._eta[key] = self._algorithm._compute_step_size(key)
            curr_grad = self._algorithm._get_gradient(key)

            grad.append(curr_grad)

            try:
                self._param_slices[key]
            except:
                first_index = self._param_length
                last_index = first_index + curr_grad.shape[0]
                self._param_slices[key] = slice(first_index, last_index)
                self._param_length += curr_grad.shape[0]

            # NEW (MATLAB Equivalent):
            old_data = self._algorithm._data[key][:, 1:]
            
            # Keep internal buffer aligned with the model's input-indexed local view.
            new_col = inputData[key].reshape(-1, 1)
            if new_col.shape[0] != old_data.shape[0]:
                in_idx = self._algorithm._in_idx.get(key, [])
                if in_idx:
                    new_col = new_col[np.asarray(in_idx, dtype=int), :]
            
            self._algorithm._data[key] = np.hstack([old_data, new_col])

        grad = np.vstack(grad).flatten()   
        return grad
    

    def update_params(self, update_term):
        # super().update_params(update_term=update_term)
        if len(update_term.shape) == 1:
                
                update_term = np.reshape(update_term, shape = (update_term.shape[0], 1))
        new_param = np.zeros_like(update_term)
        for key in self._algorithm._data_keys:
            param_slice = self._param_slices[key]
            new_param[param_slice] = self._params[param_slice] + self._eta[key] * update_term[param_slice]

            # if len(update_term.shape) == 1:
                    
            #         update_term = np.reshape(update_term, shape = (update_term.shape[0], 1))

            # try:
            #     self._params += update_term
            # except:
            #     import pdb; pdb.set_trace()
        self.set_params(new_params = new_param)


    
    def set_params(self, new_params):
        super().set_params(new_params)
        # import pdb; pdb.set_trace()
        for key in self._algorithm._data_keys:
            param_slice = self._param_slices[key]
            new_param = new_params[param_slice].reshape(-1,1)
            # new_param = self._algorithm.soft_threshold(self._eta[key], new_param, self._algorithm._lambda)
            self._algorithm._theta[key] = new_param
        self._params = new_params

        # print(f"New Params: {new_param}")

    def estimate(self, input_data, **kwargs):
        return self._algorithm.forecast(steps=kwargs.get("steps", 1))


class CCVARPartialInModel(CCVARPartialModel):
    def __init__(self, algorithmParam, cellularComplex):
        ccvarParams = (algorithmParam, cellularComplex)
        algorithm = CCVARPartialIn(*ccvarParams)
        initial_params = []
        for key in algorithm._data_keys:
            initial_params.append(algorithm._theta[key])
        initial_params = np.vstack(initial_params)

        BaseModel.__init__(self, initial_params=initial_params, algorithm=algorithm)
        self._param_slices = dict()
        self._param_length = 0
        self._eta = dict()


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

    # @property
    # def Hodge_Laplacian(self):
    #     return self

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
        # super().update_params(update_term=update_term)
        if len(update_term.shape) == 1:
                
                update_term = np.reshape(update_term, shape = (update_term.shape[0], 1))
        new_param = np.zeros_like(update_term)
        for key in self._algorithm._data_keys:
            param_slice = self._param_slices[key]
            new_param[param_slice] = self._params[param_slice] + self._eta[key] * update_term[param_slice]

            # if len(update_term.shape) == 1:
                    
            #         update_term = np.reshape(update_term, shape = (update_term.shape[0], 1))

            # try:
            #     self._params += update_term
            # except:
            #     import pdb; pdb.set_trace()
        self.set_params(new_params = new_param)


    
    def set_params(self, new_params):
        super().set_params(new_params)
        # import pdb; pdb.set_trace()
        for key in self._algorithm._data_keys:
            param_slice = self._param_slices[key]
            new_param = new_params[param_slice].reshape(-1,1)
            # new_param = self._algorithm.soft_threshold(self._eta[key], new_param, self._algorithm._lambda)
            self._algorithm._theta[key] = new_param
        self._params = new_params

        # print(f"New Params: {new_param}")

    def estimate(self, input_data, **kwargs):
        return self._algorithm.forecast(steps = kwargs.get('steps', 1))
        # return self._algorithm._data
        # return self._algorithm._theta




    
    
