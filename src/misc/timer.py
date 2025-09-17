# timer.py
import time

class _Section:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
    def __enter__(self):
        self.timer._enter(self.name)
    def __exit__(self, exc_type, exc, tb):
        self.timer._exit(self.name)

class PhaseTimer:
    """
    Minimal block-level profiler for 'named phases'.
    Example report:
      rollout            12.34s  45.6%
      forward+loss        8.90s  32.9%
      backward            4.21s  15.6%
      optim.step()        1.06s   3.9%
      logging             0.47s   1.7%
    """
    def __init__(self, use_cuda=False, device=None):
        self.totals = {}
        self._stack = []
        self.use_cuda = use_cuda
        self.device = device

    def section(self, name: str):
        """Use as:  with timer.section('rollout'): ..."""
        return _Section(self, name)

    def _sync(self):
        if self.use_cuda:
            import torch
            torch.cuda.synchronize(self.device)

    def _enter(self, name):
        self._sync()
        self._stack.append((name, time.perf_counter()))

    def _exit(self, name):
        self._sync()
        n, t0 = self._stack.pop()
        assert n == name, f"Mismatched section nesting: {n} vs {name}"
        dt = time.perf_counter() - t0
        self.totals[name] = self.totals.get(name, 0.0) + dt

    def report(self, *, iterations=None, sort=True, file=None):
        total = sum(self.totals.values())
        rows = []
        for name, dur in self.totals.items():
            pct = (dur / total * 100.0) if total else 0.0
            per_iter = (dur / iterations) if iterations else None
            rows.append((name, dur, pct, per_iter))
        if sort:
            rows.sort(key=lambda r: r[1], reverse=True)

        out = []
        header = f"{'phase':20} {'time':>10} {'%':>7}" + (f" {'/iter':>10}" if iterations else "")
        out.append(header)
        out.append("-" * len(header))
        for name, dur, pct, per_iter in rows:
            line = f"{name:20} {dur:10.2f}s {pct:6.1f}%"
            if iterations:
                line += f" {per_iter:10.4f}s"
            out.append(line)
        text = "\n".join(out)
        print(text, file=file or None)
        return text
