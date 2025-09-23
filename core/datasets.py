# registry/datasets.py
_DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(fn):
        _DATASET_REGISTRY[name] = fn
        return fn
    return decorator

def get_dataset(name, **kwargs):
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found in registry!")
    return _DATASET_REGISTRY[name](**kwargs)
