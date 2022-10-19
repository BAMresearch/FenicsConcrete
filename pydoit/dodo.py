import pathlib

DOIT_CONFIG = {
    "action_string_formatting": "both",
    "verbosity": 2,
}

parent_dir_dodo = pathlib.Path(__file__).parent
parent_dir_test = parent_dir_dodo.parent / "test"

# print(parent_dir_test)
#'actions': [f"python {test_file}"],         #'actions': ["python %s" %test_file],

def task_hello():
    """hello"""
    test_file = parent_dir_test / "example.py"
    print(test_file)
    return {
        'actions': ["python {}".format(test_file)],
        'verbosity' : 2
        }

