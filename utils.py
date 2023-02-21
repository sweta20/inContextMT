
DOMAIN_DATA_DIR = "multi-domain"

def read_file(fname, transform=lambda x: x):
    data = []
    with open(fname) as f:
        for line in f:
            data.append(transform(line.strip()))
    return data


def convert_input_to_template(prompts):
    return (" </s> ").join([f"{prompt.data['src']} = {prompt.data['tgt']}" for prompt in prompts])


def get_data(domain, src_lang, tgt_lang, split):
    src = read_file(f"{DOMAIN_DATA_DIR}/{domain}/{split}.{src_lang}")
    tgt = read_file(f"{DOMAIN_DATA_DIR}/{domain}/{split}.{tgt_lang}")
    return src, tgt


class FewShotSample(object):
    def __init__(
        self,
        data,
        correct_candidates=None,
    ):
        self._data = data
        self._correct_candidates = correct_candidates

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, item):
        return item in self._data

    @property
    def correct_candidates(self):
        return self._correct_candidates

    def is_correct(self, candidate):
        return candidate in self.correct_candidates

    @property
    def data(self):
        return self._data
