"""

Base Distributed Optimization Class

"""

class BaseModel:

    def __init__(self, algClass, algParams):
        # Initialize the model
        self._algInstance = algClass(algParams)

    def update_gradient(self, **kwargs):
        pass

    