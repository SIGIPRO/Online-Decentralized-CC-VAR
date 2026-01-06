"""

DistributedAgent:

Agent class to hold neighborhood reports, exchange information between agents, and apply distributed optimization.

If one thinks about the control/RL framework, the followings are states and actions, and rewards:

STATES: Neighborhood reports
ACTIONS: Aggregation of local information, apply gradients, send information to others
REWARD: Reward is the error achieved by the Model


"""
class BaseAgent:

    def __init__(self, cluster_id, model, data, protocol, mix, imputer, neighbors):

        self._cluster_id = cluster_id

        self._data = data
        self._model = model
        self._protocol = protocol
        self._mixing_model = mix
        self._imputer = imputer
        self._neighbor_clusters = neighbors
        self._t = 0

    def update(self, **kwargs):
        """
        One discrete time step in the life of the agent.
        """
        t = kwargs.get("t")
        if t is None:
            t = self._t
            kwargs = {**kwargs, "t": t}
        try: 
            next(self._data)  # Advance data iterator if applicable
        except StopIteration:
            return False

        # 1. DATA PHASE (K_data)
        # ----------------------
        # Send local measurements to neighbors who need them
        if self._protocol.should_send_data(t):
            for cluster_id in self._neighbor_clusters:
                # We send data relevant to the boundary with 'cluster_id'
                exporter = getattr(self._data, "export_data", None)
                payload = exporter(target=cluster_id) if callable(exporter) else self._data
                self._protocol.send_data(payload, target=cluster_id)

        incoming_data_map = self._receive_data()
        
        # if incoming_data_map:
        #     # self._data = self._append_data(self._data, incoming_data_map)
        #     self._data.append_data(incoming_data_map)

        self._data = self._imputer.impute(self._data, incoming_data_map)

        # 2. GRADIENT PHASE
        # -----------------
        # Model computes gradient based on D_in and D_out_stale
        # This covers Eq [write the equation number or name here if applicable]
        local_grad = self._model.get_gradient(self._data, **kwargs)

        # 3. PARAMETER PHASE (K_param)
        # ---------------------------
        if self._protocol.should_send_params(t):
            for cluster_id in self._neighbor_clusters:
                self._protocol.send_params(self._model.get_params(), target=cluster_id)

        # Collect params from ALL neighbors for the Mixing Model
        neighbor_params = {}
        for cluster_id in self._neighbor_clusters:
            p = self._protocol.receive_params(cluster_id)
            if p is not None:
                neighbor_params[cluster_id] = p

        # 4. MIXING & UPDATING
        # --------------------
        # The mixing model applies Gradient Tracking / Consensus logic
          # 5. Correction & Set New Params

        new_theta = self._model.get_params() - self._mixing_model.apply_correction(local_gradient=local_grad)
        self._model.set_params(new_theta)

        if neighbor_params != {}:
            self._mixing_model.update_auxiliary(local_params=self._model.get_params(),
                                            neighbor_params_dict=neighbor_params)
            new_theta = self._mixing_model.mix_parameters(local_params=self._model.get_params(),
                                            neighbor_params_dict=neighbor_params)
            self._model.set_params(new_theta)
    
   
        
      
        self._t = t + 1

        return True

    # 
    def estimate(self, **kwargs):
        return self._model.estimate(**kwargs)
    
    def _receive_data(self):
          
        incoming_data_map = {}
        
        for cluster_id in self._neighbor_clusters:
            data = self._protocol.receive_data(cluster_id)
            if data is not None:
                incoming_data_map[cluster_id] = data

        return incoming_data_map

    
