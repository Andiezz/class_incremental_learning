import os
import importlib


def get_all_models():
    return [
        model.split(".")[0]
        for model in os.listdir("models")
        if model.find("__") <= -1 and "py" in model
    ]


names = {}
for model in get_all_models():
    mod = importlib.import_module("models." + model)
    model_key = model.replace("_", "")
    class_dict = {x.lower(): x for x in mod.__dir__() if hasattr(mod, x) and not x.startswith("__")}
    if model_key in class_dict:
        class_name = class_dict[model_key]
        names[model] = getattr(mod, class_name)
    else:
        print(f"Warning: Could not find class for model '{model}'")


def get_model(args, backbone, loss, transform=None):
    return names[args.model](backbone, loss, args, transform)
