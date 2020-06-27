from argus.metrics import Metric
from cnd.ocr.converter import strLabelConverter
from Levenshtein import distance
import torch



class StringAccuracy(Metric):
    name = "str_accuracy"
    better = "max"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        if isinstance(targets, list):
            targets = ''.join(targets)
            targets = self.encoder.encode(targets)[0]

        encoded_preds = self.encoder.encode(preds)[0]
        min_len = min(len(encoded_preds), len(targets))

        for i in range(min_len):
            if encoded_preds[i] == targets[i]:
                self.correct += 1
            self.count += 1

    def compute(self):
        if self.count == 0:
            # raise Exception('Must be at least one example for computation')
            return 0
        return self.correct / self.count

class LevDistance(Metric):
    name = "str_levenshtein"
    better = "min"

    def __init__(self):
        self.alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"
        self.encoder = strLabelConverter(self.alphabet)

    def reset(self):
        self.levdist = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        if isinstance(targets, list):
            targets = ''.join(targets)
            targets = self.encoder.encode(targets)[0]

        decoded_targets = self.encoder.decode(targets, torch.IntTensor([len(targets)]))
        preds_joined = ''.join(preds)
        min_len = min(len(preds), len(targets))
        self.levdist = distance(preds_joined, decoded_targets) / min_len

    def compute(self):
        return self.levdist
