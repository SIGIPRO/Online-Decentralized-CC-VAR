from src.core import BaseImputer
import numpy as np


class LabelPropagator(BaseImputer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lambda = kwargs['lambda']

    def impute(self, current_data, incoming_data_map, metadata=None):
        if not isinstance(incoming_data_map, dict):
            return current_data
        data_map = {}

        if metadata == None:
            return current_data
        
        try: self._dimensions
        except: 
            self._dimensions = list(metadata.keys())
            self._inv_lap = dict()

            for dim in self._dimensions:
                
                B_l = metadata.get(dim - 1, 0)
                B_u = metadata.get(dim, 0)

                self._inv_lap[dim] = B_u @ B_u.T + B_l.T @ B_l

                np.fill_diagonal(self._inv_lap[dim], self._lambda[dim] * self._inv_lap[dim].diagonal() + 1)

                self._inv_lap[dim] = np.linalg.pinv(self._inv_lap[dim])
                



        incoming_neighbors = set(incoming_data_map.keys())
        state_neighbors = set(self._state.keys()) - incoming_neighbors
        
        for cluster_id in incoming_neighbors:
            payload = incoming_data_map.get(cluster_id, None)
            if payload is not None:
                # self._state[cluster_id] = payload
                data_map[cluster_id] = payload
            else:
                state_neighbors.update({cluster_id})
                incoming_neighbors -= {cluster_id}
        current_data.append_data(data_map, accept_stale = True)
        
        y = current_data.data
        ## Fill y
        for cluster_id in state_neighbors:
           if 'ogidx' not in self._state[cluster_id]:
                    continue
           
           for dim in self._state[cluster_id]['ogidx']:
               i = 0
               for gidx in self._state[cluster_id]['ogidx'][dim]:
                   dataIdx = current_data._global_idx[dim].index(gidx)
                   y[dim][dataIdx] = self._state[cluster_id]['ogidx'][dim][i]

                   i+= 1

           if 'igidx' not in self._state[cluster_id]:
                    continue
           
           for dim in self._state[cluster_id]['igidx']:
               i = 0
               for gidx in self._state[cluster_id]['igidx'][dim]:
                   dataIdx = current_data._global_idx[dim].index(gidx)
                   y[dim][dataIdx] = self._state[cluster_id]['igidx'][dim][i]

                   i+= 1
        
        x = self._inv_lap @ y
        propagate_map = dict()

        ## Fill propagate_map
        for cluster_id in state_neighbors:
           if 'ogidx' not in self._state[cluster_id]:
                    continue
           
           if cluster_id not in propagate_map:
                propagate_map[cluster_id] = dict()
                propagate_map[cluster_id]['ogidx'] = self._state[cluster_id]['ogidx']
           
           for dim in propagate_map[cluster_id]['ogidx']:
               
               local_idx = [current_data._global_idx[dim].index(ogidx) for ogidx in propagate_map['ogidx'][dim]]

               propagate_map['odata'][dim] = x[np.array(local_idx)]

           if 'igidx' not in self._state[cluster_id]:
                    continue
           
           propagate_map[cluster_id]['igidx'] = self._state[cluster_id]['igidx']
           
           for dim in self._state[cluster_id]['igidx']:
               local_idx = [current_data._global_idx[dim].index(ogidx) for ogidx in propagate_map['igidx'][dim]]

               propagate_map['idata'][dim] = x[np.array(local_idx)]


            ##Fill state with the incoming data map if applicable

           for cluster_id in incoming_neighbors:
                self._state[cluster_id] = incoming_data_map[cluster_id]
           
           


        
        current_data.append_data(propagate_map, accept_stale = True)
        return current_data
        