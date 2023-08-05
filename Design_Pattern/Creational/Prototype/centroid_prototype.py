class CentroidPrototype:
    def __init__(self, data):
        self.data = data

    def clone(self):
        return CentroidPrototype(self.data)
