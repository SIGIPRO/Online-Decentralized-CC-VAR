from src.core import BaseImputer

class DataHolder(BaseImputer):

    ## TODO: ChatGPT wrote the impute wrong.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def impute(self, current_data, incoming_data_map, metadata=None):
        incoming_flags = {}
        if isinstance(metadata, dict):
            incoming_flags = metadata.get("incoming", {}) or {}

        # if incoming_data_map is None:
        #     incoming_data_map = {}
        if not isinstance(incoming_data_map, dict):
            return current_data

        # last_incoming = self._state or {}
        data_map = {}

        neighbors = set(incoming_flags.keys()) & set(incoming_data_map.keys()) & set(self._state.keys())
        for cluster_id in neighbors:
            if incoming_flags.get(cluster_id, False):
                payload = incoming_data_map.get(cluster_id)
                if payload is not None:
                    self._state[cluster_id] = payload
                    data_map[cluster_id] = payload
            else:
                payload = self._state.get(cluster_id)
                if payload is not None:
                    data_map[cluster_id] = payload

        # self._state["last_incoming"] = last_incoming
        if data_map:
            current_data.append_data(data_map)
        return current_data
