import importlib


def find_dataset_generator_using_name(dataset_generator_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    dataset_filename = "dataset." + dataset_generator_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset_generator = None
    target_dataset_name = dataset_generator_name + '_dataset'
    for name, potential_model_creator in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset_generator = potential_model_creator

    if dataset_generator is None:
        print("In %s.py, there should be a func with name that matches %s in lowercase." % (
            dataset_filename, target_dataset_name))
        exit(0)

    return dataset_generator


if __name__ == '__main__':
    model = find_dataset_generator_using_name('no_gene_seq')()
    print(model)
