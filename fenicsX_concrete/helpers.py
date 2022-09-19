import dolfinx as df


class Parameters(dict):
    """
    Dict that also allows to access the parameter
        p["parameter"]
    via the matching attribute
        p.parameter
    to make access shorter
    """
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def __add__(self, other):
        if other == None:
            dic = self
        else:
            dic = Parameters({**self, **other})
        return dic

