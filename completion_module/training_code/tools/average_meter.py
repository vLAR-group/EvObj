class AverageMeter:
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0.0] * self.n_items
        self._sum = [0.0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if not isinstance(values, list):
            values = [values]
        for i, v in enumerate(values):
            self._val[i] = v
            self._sum[i] += v
            self._count[i] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val if self.n_items > 1 else self._val[0]
        return self._val[idx]

    def avg(self, idx=None):
        if idx is None:
            if self.n_items == 1:
                return self._sum[0] / max(self._count[0], 1)
            return [self._sum[i] / max(self._count[i], 1) for i in range(self.n_items)]
        return self._sum[idx] / max(self._count[idx], 1)

    def count(self, idx=None):
        if idx is None:
            return self._count if self.n_items > 1 else self._count[0]
        return self._count[idx]
