from __future__ import annotations

from src.core.agent import BaseAgent


class SnapshotAgent(BaseAgent):
    """
    Agent variant for shared-snapshot experiments.
    - Data snapshot is shared across all agents.
    - No data communication/imputation.
    - Parameter communication/mixing is unchanged.
    """

    _shared_data_stream = None

    @classmethod
    def set_shared_data_stream(cls, data_stream):
        cls._shared_data_stream = data_stream

    @classmethod
    def reset_shared_data_stream(cls):
        cls._shared_data_stream = None

    @classmethod
    def advance_shared_snapshot(cls):
        if cls._shared_data_stream is None:
            raise ValueError("SnapshotAgent shared data stream is not set.")
        try:
            next(cls._shared_data_stream)
            return True
        except StopIteration:
            return False

    def __init__(self, cluster_id, model, protocol, mix, neighbors):
        self._cluster_id = cluster_id
        self._model = model
        self._protocol = protocol
        self._mixing_model = mix
        self._neighbor_clusters = neighbors
        self._neighbor_params = {}
        self._neighbor_aux = {}
        self._t = 0

    def iterate_data(self, t):
        self._t = t
        return True

    def send_data(self, t):
        del t
        return

    def receive_data(self):
        return

    def local_step(self):
        shared_data = type(self)._shared_data_stream
        if shared_data is None:
            raise ValueError("SnapshotAgent shared data stream is not set before local_step.")

        local_grad = self._model.get_gradient(aggregated_data=shared_data)
        update_term = self._mixing_model.apply_correction(local_gradient=local_grad)
        self._model.update_params(update_term=-update_term)

