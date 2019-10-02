import chainer
from chainer import function
from chainer import reporter as reporter_module
import copy


class MyEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, target, converter, device):
        super(MyEvaluator, self).__init__(iterator, target, converter, device)

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device, padding=-1)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()
