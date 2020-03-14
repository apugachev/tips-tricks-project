from argus.metrics import Metric


class StringAccuracy(Metric):
    name = "str_accuracy"
    better = "max"

    def reset(self):
        self.correct = 0
        self.count = 0

    def update(self, step_output: dict):
        preds = step_output["prediction"]
        targets = step_output["target"]
        # TODO: Count correct answers

    def compute(self):
        if self.count == 0:
            # raise Exception('Must be at least one example for computation')
            return 0
        return self.correct / self.count

# TODO: In the same way you can write Accuracy by position of letter
# or quality of negative examples and target
