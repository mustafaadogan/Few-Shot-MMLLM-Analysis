class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, model_class_methods):
        self.models[name] = model_class_methods

model_registry = ModelRegistry()