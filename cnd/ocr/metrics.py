import torch.nn as nn
from cnd.ocr.converter import strLabelConverter
import torch
from catalyst.dl.core import MultiMetricCallback
from typing import List
from sklearn.metrics import accuracy_score as acc
from Levenshtein import distance

class WrapCTCLoss(nn.Module):
    def __init__(self, alphabet):
        super().__init__()
        self.converter = strLabelConverter(alphabet)
        self.loss = nn.CTCLoss()

    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)
        return sim_preds, preds_size

    def __call__(self, logits, targets):
        text, length = self.converter.encode(targets)
        sim_preds, preds_size = self.preds_converter(logits, len(targets))
        loss = self.loss(logits, text, preds_size, length)
        return loss

def _get_default_accuracy_args(num_classes: int) -> List[int]:

    result = [1]

    if num_classes is None:
        return result

    if num_classes > 3:
        result.append(3)
    if num_classes > 5:
        result.append(5)

    return result


class WrapAccuracyScore(MultiMetricCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
        num_classes: int = None
    ):
        list_args = accuracy_args or _get_default_accuracy_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=self.__myaccuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key
        )

    def __myaccuracy(self, outputs, targets, list_args):
        alphabet = " ABEKMHOPCTYX"
        alphabet += "".join([str(i) for i in range(10)])

        ctc_wrap = WrapCTCLoss(alphabet)
        preds, preds_size = ctc_wrap.preds_converter(outputs, outputs.shape[1])
        return [acc(targets, preds)]

class WrapLevenshteinScore(MultiMetricCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "levenshtein",
        accuracy_args: List[int] = None,
        num_classes: int = None
    ):
        list_args = accuracy_args or _get_default_accuracy_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=self.__mylevenshtein,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key
        )

    def __mylevenshtein(self, outputs, targets, list_args):
        alphabet = " ABEKMHOPCTYX"
        alphabet += "".join([str(i) for i in range(10)])

        ctc_wrap = WrapCTCLoss(alphabet)
        preds, _ = ctc_wrap.preds_converter(outputs, outputs.shape[1])
        levs = [distance(t,p) for t,p in zip(targets, preds)]
        return levs




#TODO: ADD ACCURACY https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/callbacks/metrics/accuracy.html
# YOU WILL NEED TO WRAP STANDARD ACCURACY, AS CTCLOSS ABOVE
# https://github.com/catalyst-team/catalyst-info#catalyst-info-5-callbacks