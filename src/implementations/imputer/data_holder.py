from src.core import BaseImputer

class DataHolder(BaseImputer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def impute(self, current_data, incoming_data_map, metadata=None):
        
        if not isinstance(incoming_data_map, dict):
            return current_data
        data_map = {}

        neighbors = set(self._state.keys()) | set(incoming_data_map.keys())
        for cluster_id in neighbors:
            payload = incoming_data_map.get(cluster_id, None)
            if payload is not None:
                self._state[cluster_id] = payload
                data_map[cluster_id] = payload
            else:
                payload = self._state.get(cluster_id, None)
                if payload is not None:
                    data_map[cluster_id] = payload

        if data_map:
            current_data.append_data(data_map, accept_stale = True)
        return current_data
