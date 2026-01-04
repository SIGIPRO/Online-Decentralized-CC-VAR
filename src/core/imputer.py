class BaseImputer:
    """
    Base class for data imputation strategies.

    Imputers take local data plus any incoming neighbor data and return
    an updated data object that the model can consume.
    """

    def __init__(self, **kwargs):
        self._state = {}

    def impute(self, current_data, incoming_data_map, t=None, metadata=None):
        """
        Apply the imputation strategy.

        Args:
            current_data: Local data container (e.g., CellularComplexInMemoryData).
            incoming_data_map: Data received from neighbors, if any.
            t: Optional time step.
            metadata: Optional dict (e.g., topology, masks).
        Returns:
            Updated data object/structure.
        """
        raise NotImplementedError("Define the imputation rule in a derived class.")

    def reset(self):
        """Clear any cached state for a new run."""
        self._state.clear()
