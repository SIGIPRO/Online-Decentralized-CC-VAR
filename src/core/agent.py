class BaseAgent:

    def __init__(self, cluster_id, model, data, protocol, mix, imputer, neighbors, cellularComplex):

        self._cluster_id = cluster_id

        self._data = data
        self._model = model
        self._protocol = protocol
        self._mixing_model = mix
        self._imputer = imputer
        self._neighbor_clusters = neighbors
        self._cellularComplex = cellularComplex
        self._neighbor_params = {}
        self._neighbor_aux = {}
        self._t = 0

    def iterate_data(self, t):
        try: 
            next(self._data)
            self._t = t # Advance data iterator if applicable
        except StopIteration:
            return False
    
    def send_data(self, t):
         if self._protocol.should_send_data(t):
            for cluster_id in self._neighbor_clusters:
                # We send data relevant to the boundary with 'cluster_id'
                exporter = getattr(self._data, "export_data", None)
                payload = exporter(target=cluster_id) if callable(exporter) else self._data
                self._protocol.send_data(payload, target=cluster_id)

    def send_params(self, t):
         if self._protocol.should_send_params(t):
            for cluster_id in self._neighbor_clusters:
                parameter_dict = dict()
                parameter_dict['aux'] = self._mixing_model.get_aux()
                parameter_dict['params'] = self._model.get_params()
                self._protocol.send_params(parameter_dict, target=cluster_id)


    def receive_data(self):
        incoming_data_map = {}
        
        for cluster_id in self._neighbor_clusters:
            data = self._protocol.receive_data(cluster_id)
            if data is not None:
                incoming_data_map[cluster_id] = data

        self._data = self._imputer.impute(self._data,
                                        incoming_data_map = incoming_data_map,
                                        metadata = self._cellularComplex)

    def prepare_params(self, t):
        if not self._protocol.should_send_params(t):
            return
        else:
            self._mixing_model.update_aux(local_params = self._model.get_params())
        

    def receive_params(self):
        ## Reset the parameters and auxilary variables
        self._reset_neighbor_params()
     
        for cluster_id in self._neighbor_clusters:
            p = self._protocol.receive_params(cluster_id)
            if p is not None:
                self._neighbor_params[cluster_id] = p['params']
                self._neighbor_aux[cluster_id] = p['aux']
            
    def _reset_neighbor_params(self):
        self._neighbor_params = {}
        self._neighbor_aux = {}

    def do_consensus(self):
        self._model.set_params(
            self._mixing_model.mix_parameters(
                neighbor_params = self._neighbor_params,
                neighbor_aux = self._neighbor_aux
            )
        )

    @property
    def outbox(self):
        return self._protocol.flush_outboxes()
    
  
    def push_to_inbox(self, source, payload, msg_type):
        self._protocol.push_to_inbox(source, payload, msg_type)

    def local_step(self):
        local_grad = self._model.get_gradient(aggregated_data = self._data)
        if self._cluster_id == 0 and self._t == 360:
            import pdb; pdb.set_trace()
        update_term = self._mixing_model.apply_correction(local_gradient = local_grad)

        # import pdb; pdb.set_trace()
        self._model.update_params(update_term = update_term)

    def estimate(self, **kwargs):
        return self._model.estimate(**kwargs)

    
