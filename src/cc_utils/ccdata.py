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
            data = self._data[key]
            if data.ndim < 2:
                continue
            if T is None:
                T = data.shape[1]
            else:
                T = min(T, data.shape[1])
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
    def __init__(self, data, interface, Nout, Nex, global_idx):
        super().__init__(data)

        self._interface = interface
        self._global_idx = global_idx
        self._Nex = Nex
        
        self._get_partial_data( Nout = Nout)


    def _get_partial_data(self, Nout):
        self._interface_data = dict()
        self._curr_interface_data = dict()

        for cluster_head in Nout:
            if cluster_head not in self._interface:
                continue
            if cluster_head not in self._interface_data:
                self._interface_data[cluster_head] = dict()
            for key in self._data:
                if key not in Nout[cluster_head] or key not in self._interface[cluster_head]:
                    continue
                out_slices = [self._global_idx[key].index(gidx) for gidx in Nout[cluster_head][key]]
                interface_slices = [self._global_idx[key].index(gidx) for gidx in self._interface[cluster_head][key]]
                
                self._interface_data[cluster_head][key] = np.array([self._data[key][ifs, :] for ifs in interface_slices])

                self._data[key][out_slices, :] = 0
                self._data[key][interface_slices, :] = 0


    def export_data(self, target):
        if target not in self._Nex:
            return None
        if target not in self._interface:
            return None
        outgoing_data = dict()
        outgoing_data['t'] = self._curr_iteration
        outgoing_data['ogidx'] = self._Nex[target]

        outgoing_data['odata'] = dict()

        for key in outgoing_data['ogidx']:
            outgoing_data['odata'][key] = self._current_data[key][outgoing_data['ogidx'][key]]
        outgoing_data['igidx'] = dict()
        outgoing_data['idata'] = dict()
        for key in self._interface[target]:
            outgoing_data['igidx'][key] = self._interface[target][key]
            outgoing_data['idata'][key] = self._curr_interface_data[target][key]

        return outgoing_data

    def __next__(self):
        for cluster_head in self._interface_data:
            if cluster_head not in self._curr_interface_data:
                self._curr_interface_data[cluster_head] = dict()
            for key in self._interface_data[cluster_head]:
                
                self._curr_interface_data[cluster_head][key] = self._interface_data[cluster_head][key][:, self._curr_iteration]
        super().__next__()

    
    
    def append_data(self, incoming_data_map):
        for cluster_id in incoming_data_map:
            incoming_data = incoming_data_map.get(cluster_id, None)

            if incoming_data is None:
                continue

            if incoming_data.get('t') != self._curr_iteration:
                continue

            for key in self._current_data:
                i = 0
                if 'ogidx' not in incoming_data or key not in incoming_data['ogidx']:
                    continue
                for gidx in incoming_data['ogidx'][key]:
                    dataIdx = self._global_idx[key].index(gidx)
                    self._current_data[key][dataIdx] = incoming_data['odata'][key][i]
                    i += 1

                i = 0
                if 'igidx' not in incoming_data or key not in incoming_data['igidx']:
                    continue
                for gidx in incoming_data['igidx'][key]:
                    dataIdx = self._global_idx[key].index(gidx)

                    try: intIdx= self._interface[cluster_id][key].index(gidx)
                    except: continue

                    self._current_data[key][dataIdx] = 0.5 * (incoming_data['idata'][key][i] + self._curr_interface_data[cluster_id][key][intIdx])
                    i += 1


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
