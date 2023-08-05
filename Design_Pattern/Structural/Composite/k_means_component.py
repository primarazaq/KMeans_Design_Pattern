class K_MeansComponent:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def remove_component(self, component):
        self.components.remove(component)

    def fit(self, data):
        for component in self.components:
            component.fit(data)

    def pred(self, data):
        predictions = []
        for component in self.components:
            prediction = component.pred(data)
            predictions.append(prediction)
        return predictions
