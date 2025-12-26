from src.core.agent import BaseAgent

class ZeroPadderAgent(BaseAgent):

    def __init__(self, Nin, Nout, initial_data, protocol, protocolParams, zero_data= None):
        # Initialize with no model and no mixing
        super().__init__(model=None, modelParams={}, 
                         Nin=Nin, Nout=Nout, 
                         data=initial_data, 
                         dataParams={},
                         protocol=protocol, protocolParams=protocolParams, 
                         mix=None, mixingParams={})
        self._incame_data_map = {cluster_id: zero_data for cluster_id in self._neighbor_clusters}
        self._zero_data = zero_data
        
    def _receive_data(self):
        incoming_data_map = super()._receive_data()
        # Receive incoming measurements to fill N_out buffer
        if incoming_data_map == {}:
            incoming_data_map = {cluster_id: self._zero_data for cluster_id in self._neighbor_clusters}
        return incoming_data_map
