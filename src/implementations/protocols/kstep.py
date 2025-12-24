from src.core import BaseProtocol

class KStepProtocol(BaseProtocol):
    """
    Implements a synchronous K-step communication protocol.
    Supports independent clocks for data measurements and model parameters.
    """
    def __init__(self, K_data, K_param, heuristic=None):
        """
        Args:
            K_data (int): Frequency of measurement exchange (t % K_data == 0).
            K_param (int): Frequency of parameter exchange (t % K_param == 0).
            heuristic (callable, optional): A function to override K-step logic 
                                            based on local state or error.
        """
        super().__init__(K_data, K_param, heuristic)
        
        # Local Inboxes (filled by the Environment/Router)
        self._inbox_data = {}   # {source_agent_id: measurement_data}
        self._inbox_params = {} # {source_agent_id: model_parameters}

    # --- Timing Logic ---

    def should_send_data(self, t, local_state=None):
        """Checks if the current time instant is a data communication round."""
        if self._heuristic:
            # If a heuristic is provided, it can override the K-step logic
            return self._heuristic(t, "data", self._K_data, local_state)
        return t % self._K_data == 0

    def should_send_params(self, t, local_state=None):
        """Checks if the current time instant is a parameter mixing round."""
        if self._heuristic:
            return self._heuristic(t, "params", self._K_param, local_state)
        return t % self._K_param == 0

    # --- Outbound Logic (Agent calls these) ---

    def send_data(self, data, target):
        """Queues measurement data to be sent to a specific neighbor."""
        self._send_data[target] = data

    def send_params(self, params, target):
        """Queues model parameters to be sent to a specific neighbor."""
        self._send_params[target] = params

    # --- Inbound Logic (Agent calls these) ---

    def receive_data(self, source):
        """
        Retrieves data from a specific neighbor's inbox. 
        Returns None if no new data has arrived since last check.
        """
        return self._inbox_data.pop(source, None)

    def receive_params(self, source):
        """
        Retrieves parameters from a specific neighbor's inbox.
        Returns None if no new params have arrived.
        """
        return self._inbox_params.pop(source, None)

    # --- System Logic (Environment/Transport Layer calls these) ---

    def push_to_inbox(self, source, payload, msg_type):
        """
        Method for the Environment to deliver a message to this protocol.
        """
        if msg_type == "data":
            self._inbox_data[source] = payload
        elif msg_type == "params":
            self._inbox_params[source] = payload

    def flush_outboxes(self):
        """
        Returns all queued messages and clears the outboxes.
        Used by the Environment to route messages to neighbors.
        """
        outgoing = {
            "data": dict(self._send_data),
            "params": dict(self._send_params)
        }
        self._send_data.clear()
        self._send_params.clear()
        return outgoing