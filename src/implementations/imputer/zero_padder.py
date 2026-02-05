from src.core import BaseImputer

class ZeroPadder(BaseImputer):
    def __init__(self, **kwargs):
        pass 

    
    def impute(self, current_data, incoming_data_map, metadata = None):
        if incoming_data_map:
            current_data.append_data(incoming_data_map)
        return current_data
    ## Please note that in CCIMPartialData the data is zero padded already