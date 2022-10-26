import pathlib

DOIT_CONFIG = {
    "action_string_formatting": "both",
    "verbosity": 2,
}

parent_dir_dodo = pathlib.Path(__file__).parent
parent_dir_test = parent_dir_dodo.parent / "test"

print(parent_dir_test)
#'actions': [f"python {test_file}"],         #'actions': ["python %s" %test_file],

def task_simulation_run():
    """Runnning the simulation"""
    test_file = parent_dir_test / "example_doit.py"
    #print(test_file)
    return {
        'actions': ["python {}".format(test_file)],
        }

#print(parent_dir_test)
#'actions': [f"python {test_file}"],         #'actions': ["python %s" %test_file],

def task_create_latexpdf():
    """Creation of PDF file from the TEX file."""
    return{
        'actions' : ["pdflatex -output-directory={} {}/example_doit.tex".format(parent_dir_dodo, parent_dir_dodo)],
        'file_dep' : ["{}/example_doit.tex".format(parent_dir_dodo), "{}/example_doit.dat".format(parent_dir_dodo)],
        'targets': ["{}/example_doit.pdf".format(parent_dir_dodo)],
        'verbosity' : 1
    }

