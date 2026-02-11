from src.core import BaseMixingModel
import numpy as np


class DiffusionATCModel(BaseMixingModel):

    def __init__(self, initial_aux_vars=None, weights=None, eta=None):
        super().__init__(initial_aux_vars, weights, eta)
        self._aux = None


    def mix_parameters(self, **kwargs):
        neighbor_params = kwargs.get('neighbor_params', {})
        
        new_params = 0
        for neighbor, param_val  in neighbor_params.items():
            new_params += self._weights.get(neighbor, 0) * param_val 
        
        return new_params

        
    def apply_correction(self, local_gradient):
       return local_gradient

    def update_aux(self, **kwargs):
        self._aux = None
        
