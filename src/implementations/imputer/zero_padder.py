from src.core import BaseImputer

class ZeroPadder(BaseImputer):
    def __init__(self, **kwargs):
        pass 

    
    def impute(self, current_data, **kwargs):
        return current_data
    ## Please note that in CCIMPartialData the data is zero padded already