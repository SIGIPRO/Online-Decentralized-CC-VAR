from src.core import BaseMixingModel


class KGTMixingModel(BaseMixingModel):

    def __init__(self, initial_aux_vars=None, weights=None, eta=None):
        super().__init__(initial_aux_vars, weights, eta)
        self._eta['Kc'] = self._eta['K'] * self._eta['c']
        self._eta['Kcs'] = self._eta['K'] * self._eta['c'] * self._eta['s']


    def mix_parameters(self, **kwargs):

        # self.update_auxiliary(local_params, neighbor_params_dict)

        def update_param(eta, history, tracking):
            return history - eta * tracking

        for neighbor in self._aux['tracking'].keys():
            new_params += self._weights.get(neighbor,0) * update_param(self._eta['Kcs'] , self._history.get(neighbor, 0), self._aux['tracking'][neighbor])


        self._history['self'] = new_params

        return new_params

        
    def apply_correction(self, local_gradient):
        return self._eta['c'] * (local_gradient + self._aux['correction'])

    def update_tracking(self, local_params, neighbor_tracking):
        self._aux['tracking'] = dict()
        self._aux['tracking']['self'] = 1/(self._eta['Kc']) * (local_params - self._history['self'])

        for neighbor in neighbor_tracking.keys():
            self._aux['tracking'][neighbor] = neighbor_tracking[neighbor]['tracking']
            self._history[neighbor] = neighbor_tracking[neighbor]['history']


    def update_auxiliary(self, **kwargs):
        neighbor_params_dict = kwargs.get('neighbor_params_dict', {})
        local_params = kwargs.get('local_params', 0)

        self.update_tracking(local_params=local_params,neighbor_tracking=neighbor_params_dict)
    

        self._aux['correction'] -= self._aux['tracking']['self']
        for neighbor in self._aux['tracking'].keys():
            weight = self._weights.get(neighbor, 0)
            self._aux['correction'] += weight * self._aux['tracking'][neighbor]
        

