import numpy as np
class BaseModel:
    """
    A generic Task model. 
    It defines the loss function, the local physics, and the parameters.
    """
    def __init__(self, initial_params, algorithm, **kwargs):
        self._params = initial_params
        self._state = None # To store temporal states, e.g., VAR history
        self._algorithm = algorithm

    def get_gradient(self, aggregated_data, **kwargs):
        """
        Computes the gradient of the local loss function f_i.
        
        Args:
            aggregated_data: The data points owned by the agent.
        Returns:
            A gradient vector/matrix of the same shape as self._params.
        """
        raise NotImplementedError("Define local loss gradient calculation.")

    def estimate(self, input_data, **kwargs):
        """
        The 'Forward Pass'. 
        Predicts output/state based on current parameters.
        """
        raise NotImplementedError("Define the forward prediction/inference logic.")

    def get_params(self):
        return self._params

    def set_params(self, new_params):
        self._params = new_params

    def update_params(self, update_term):
        # import pdb; pdb.set_trace()
        if len(update_term.shape) == 1:
                
                update_term = np.reshape(update_term, shape = (update_term.shape[0], 1))

        try:
            self._params += update_term
        except:
            import pdb; pdb.set_trace()

    def update_internal_state(self, new_data):
        """
        Updates hidden states, such as buffers for time-series 
        or topological indicators.
        """
        self._state = new_data


class BaseMixingModel:
    """
    A generic Coordination model.
    Handles parameter mixing, gradient tracking, and auxiliary variables.
    """
    def __init__(self, initial_aux_vars={}, weights={}, eta={}):
        """
        Args:
            initial_aux_vars: Variables like z_i (Gradient Tracking) 
                              or lambda (ADMM).
        """
        self._aux = initial_aux_vars
        self._history = {} # To store previous values for drift correction
        self._weights = weights # Graph topology weights for mixing
        self._eta = eta

    def mix_parameters(self, local_params, neighbor_params_dict):
        """
        Performs the mixing/consensus of parameters.
        
        Args:
            local_params: Current params of the agent.
            neighbor_params_dict: {neighbor_id: params} received via Protocol.
            weights: {neighbor_id: weight} defining the graph topology.
        """
        raise NotImplementedError("Define how parameters are averaged/mixed.")

    def apply_correction(self, local_gradient, **kwargs):
        """
        Applies the correction term (C_i * g_i) to the gradient.
        This is where Gradient Tracking logic (Liu 2025) usually lives.
        """
        # Generic: return local_gradient + self._aux
        raise NotImplementedError("Define how the gradient is corrected via trackers.")

    def update_aux(self, **kwargs):
        """
        Updates trackers, dual variables, or mixing history.
        """
        raise NotImplementedError("Define the update rule for aux variables.")
        
    def get_aux(self):
        return self._aux