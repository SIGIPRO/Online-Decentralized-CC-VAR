class BaseProtocol:

    def __init__(self, K_data, K_param, heuristic=None):
        self._K_data = K_data
        self._K_param = K_param
        self._heuristic = heuristic

        self._send_params = dict()
        self._send_data = dict()


    def send_data(self, data, target):
        raise NotImplementedError('This is a base class. Please implement send_data method in the derived class.')
    def receive_data(self, source):
        raise NotImplementedError('This is a base class. Please implement receive_data method in the derived class.')
    
    def send_params(self, params, target):
        raise NotImplementedError('This is a base class. Please implement send_params method in the derived class.')
    def receive_params(self, source):
        raise NotImplementedError('This is a base class. Please implement receive_params method in the derived class.')
    

    def should_send_params(self, t):
        raise NotImplementedError('This is a base class. Please implement should_send_params method in the derived class.')
    def should_send_data(self, t):
        raise NotImplementedError('This is a base class. Please implement should_send_data method in the derived class.')
    
    def set_send_params(self, K_param):
        raise NotImplementedError('This is a base class. Please implement set_send_params method in the derived class.')
    def set_send_data(self, K_data):
        raise NotImplementedError('This is a base class. Please implement set_send_data method in the derived class.')

    def communicate(self):
        raise NotImplementedError('This is a base class. Please implement communicate method in the derived class.')
    
