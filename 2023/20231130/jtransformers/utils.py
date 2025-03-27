import csv

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __iadd__(self, other):
        for k, v in self.items():
            if k in other and other[k]:
                self[k] += other[k]
            # end
        # end

        return self
    # end


# end


# Takes the file paths as arguments
def parse_csv_file_to_json(path_file_csv):
    # create a dictionary
    elements = []

    # Open a csv reader called DictReader
    with open(path_file_csv, encoding='utf-8') as file_csv:
        # with open(path_file_csv) as file_csv:
        reader_csv = csv.DictReader(file_csv, delimiter="\t")

        # Convert each row into a dictionary
        # and add it to data
        for dict_head_value in reader_csv:
            element = {}

            for head, value in dict_head_value.items():
                # print(value)
                if value and (value[0] in ["[", "{"]):
                    # element[head] = eval(value)
                    element[head] = value
                else:
                    element[head] = value

            elements.append(element)
        # end
    # end

    return elements
# end


class Batch:
    DEVICE = 'cuda'

    def __init__(self, **kwargs):
        self.kwargs = {}
        for k, v in kwargs.items():
            if v is not None and type(v) is not bool:
                self.kwargs[k] = v.to(Batch.DEVICE)
        # end
    # end

    def __call__(self):
        return self.kwargs
    # end
# end
