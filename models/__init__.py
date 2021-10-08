import importlib


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name_if_func = model_name + '_model'
    target_model_name_if_class = model_name.replace('_', '') + 'model'
    for name, potential_model_creator in modellib.__dict__.items():
        if name.lower() == target_model_name_if_func.lower() or name.lower() == target_model_name_if_class.lower():
            model = potential_model_creator

    if model is None:
        print("In %s.py, there should be a function that matches %s or class that matches %s in lowercase." % (
        model_filename, target_model_name_if_func, target_model_name_if_class))
        exit(0)

    return model


def find_hp_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name_if_func = model_name + '_model_hp'
    target_model_name_if_class = model_name.replace('_', '') + 'model_hp'
    for name, potential_model_creator in modellib.__dict__.items():
        if name.lower() == target_model_name_if_func.lower() or name.lower() == target_model_name_if_class.lower():
            model = potential_model_creator

    if model is None:
        print("In %s.py, there should be a function that matches %s or class that matches %s in lowercase." % (
        model_filename, target_model_name_if_func, target_model_name_if_class))
        exit(0)

    return model


if __name__ == '__main__':
    model = find_model_using_name('no_gene_seq_lstm')()
    print(model)
