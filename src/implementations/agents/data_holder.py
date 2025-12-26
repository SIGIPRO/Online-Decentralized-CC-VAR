from src.core.agent import BaseAgent

class DataHolderAgent(BaseAgent):

    def __init__(self, Nin, Nout, initial_data, protocol, protocolParams, incoming_data_initializer = None):
        # Initialize with no model and no mixing
        super().__init__(model=None, modelParams={}, 
                         Nin=Nin, Nout=Nout, 
                         data=initial_data, 
                         dataParams={},
                         protocol=protocol, protocolParams=protocolParams, 
                         mix=None, mixingParams={})
        self._incame_data_map = {cluster_id: incoming_data_initializer for cluster_id in self._neighbor_clusters}
        
    def _receive_data(self):
        incoming_data_map = super()._receive_data()
        # Receive incoming measurements to fill N_out buffer
        if incoming_data_map == {}:
            incoming_data_map = self._incame_data_map
        return incoming_data_map
