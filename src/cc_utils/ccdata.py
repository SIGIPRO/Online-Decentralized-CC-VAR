from scipy.io import loadmat
import numpy as np

class CellularComplexInMemoryData:

    def __init__(self, data = None):
        self._current_data = {}
        self._data = data if data is not None else {}

        self.__estimate_T()
    
    @classmethod
    def from_matlab(cls, matlab_file_path):
        matlab_data = loadmat(matlab_file_path)
        instance = cls()
        instance._data = matlab_data

        instance.__estimate_T()
        return instance
    
    ## Placeholder for future PyTorch handler
    @classmethod
    def from_pytorch(cls, torch_data):
        instance = cls()
        instance._data = torch_data
        return instance
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._T_total is None or self._T_total == 0:
            raise StopIteration
        
        for key in self._data:
            self._current_data[key] = self._data[key][:, self._curr_iteration]  # Get the first time step

        self._curr_iteration += 1
        if self._curr_iteration >= self._T_total:
            self._curr_iteration = 0  # Reset for potential future iterations
            raise StopIteration
    @property
    def data(self):
        return self._current_data

    
    def __estimate_T(self):
        T = None
        for key in self._data:
            if T is None:
                T = self._data[key].shape[1]
            else:
                T = min(T, self._data[key].shape[1])
        self._T_total = T
        self._curr_iteration = 0

    def append_data(self, incoming_data_map):
        ## !!!! CHECK THIS FUNCTION FOR CORRECTNESS !!!! There should be no index mismatch
        for cluster_id in incoming_data_map:
            incoming_data = incoming_data_map[cluster_id]
            for key in incoming_data:
                if key in self._data:
                    self._current_data[key] = np.hstack((self._current_data[key], incoming_data[key]))
                else:
                    self._current_data[key] = incoming_data[key]

## A data handler that mimics partial data sharning in CC settings. In this example, each cluster completely knows the partial data but these are exchanged when append_data is called. append_data only takes average of the partial_data inside the object and the incoming partial data. More complex strategies are left for future work.

class CCIMPartialData(CellularComplexInMemoryData):
    def __init__(self, data, relevant_cells_dict):
        super().__init__(data)
        self.relevant_cells_dict = relevant_cells_dict  # {cluster_id: [list of relevant cell indices]}
    
    # def get_relevant_data_for_cluster(self, cluster_id):
    #     relevant_indices = self.relevant_cells_dict.get(cluster_id, [])
    #     relevant_data = {}
    #     for key in self._data:
    #         relevant_data[key] = self._data[key][relevant_indices, :]
    #     return relevant_data
    
    def append_data(self, incoming_data_map):
        for cluster_id in incoming_data_map:
            incoming_data = incoming_data_map[cluster_id]
            relevant_indices = self.relevant_cells_dict.get(cluster_id, [])
            for key in incoming_data:
                if key in self._data:
                    # Average the relevant parts
                    self._data[key][relevant_indices[key], :] = 0.5 * (self._data[key][relevant_indices[key], :] + incoming_data[key][relevant_indices, :])
                else:
                    self._data[key] = incoming_data[key]
class ZeroCCData(CellularComplexInMemoryData):
    def __init__(self, shape_dict):
        data = {}
        for key in shape_dict:
            shape = shape_dict[key]
            data[key] = np.zeros(shape)
        super().__init__(data)

## Placeholder for future streaming data handler
class CellularComplexStreamingData:
    def __init__(self):
        pass