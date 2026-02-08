from src.core import BaseMixingModel
import numpy as np


class KGTMixingModel(BaseMixingModel):

    def __init__(self, initial_aux_vars=None, weights=None, eta=None):
        super().__init__(initial_aux_vars, weights, eta)
        self._eta['Kc'] = self._eta['K'] * self._eta['c']
        self._eta['Kcs'] = self._eta['K'] * self._eta['c'] * self._eta['s']
        self._aux = 0

        # import pdb; pdb.set_trace()

        if isinstance(initial_aux_vars, dict):
            self._correction = initial_aux_vars.get('correction', 0)


    def mix_parameters(self, **kwargs):
        neighbor_params = kwargs.get('neighbor_params', {})
        neighbor_aux = kwargs.get('neighbor_aux', {})

        self._correction -= self._aux
        self._correction += self._weights.get('self', 0) * self._aux
        for neighbor, aux_val in neighbor_aux.items():
            self._correction += self._weights.get(neighbor, 0) * aux_val

        def update_params(eta, history, tracking):
            update_term = history - eta * tracking
            return update_term

        new_params = self._weights.get('self', 0) * update_params(self._eta['Kcs'], self._history.get('self', 0), self._aux)
        for neighbor in neighbor_aux:
            new_params += self._weights.get(neighbor,0) * update_params(self._eta['Kcs'] , neighbor_params[neighbor], neighbor_aux[neighbor])
            # new_params += self._weights.get(neighbor,0) * update_params(self._eta['Kcs'] , self._history.get(neighbor, 0), neighbor_aux[neighbor])

        ## Update history after mixing
        self._history['self'] = new_params
        self._history.update(neighbor_params)

        return new_params

        
    def apply_correction(self, local_gradient):
        try:
            return self._eta['c'] * (local_gradient.flatten() + self._correction.flatten())
        except AttributeError:
            return self._eta['c'] * (local_gradient + self._correction)

    def update_aux(self, **kwargs):
        local_params = kwargs.get('local_params', 0)
        self._aux = 1/(self._eta['Kc']) * (- local_params + self._history.get('self', 0))
        
