class Registry:
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, key):
        def wrapper(fn_or_cls):
            if key in self._registry:
                raise ValueError(f"{key} already registered in {self.name}")
            self._registry[key] = fn_or_cls
            return fn_or_cls
        return wrapper

    def get(self, key):
        if key not in self._registry:
            raise ValueError(f"{key} not found in {self.name}. "
                             f"Available: {list(self._registry.keys())}")
        return self._registry[key]

    def list(self):
        return list(self._registry.keys())