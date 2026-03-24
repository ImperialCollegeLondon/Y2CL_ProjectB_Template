"""Decorator utilities."""
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, cast

import matplotlib.animation as mpla
import matplotlib.figure as mplf
from matplotlib.animation import ArtistAnimation
from matplotlib.figure import Figure


@dataclass
class SaveOutput:
    """save analysis output for marking."""
    plot_names: str | list[str]
    plot_output_indices: Callable[[Any], Any | Iterable[Any]] = field(default=lambda x: x, kw_only=True)
    plots_dir: Path = field(default=Path(__file__).parent.parent.parent / "plots_for_marking", kw_only=True)

    def __post_init__(self):
        if isinstance(self.plot_names, str):
            self.plot_names = [self.plot_names]
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, func: Callable[[], Any]) -> Callable[[], Any]:
        # this is now done as a patch in the conftest.py
        # if __name__ != "__main__":  # Stops saving when running tests, only when running analysis tasks interactively
        #     print(f"Patching out plot saving for: {func.__name__!r}")
        #     return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_ret = func(*args, **kwargs)
            if func_ret is None:
                return func_ret

            extracted = self.plot_output_indices(func_ret)
            if not isinstance(extracted, (list, tuple)):
                extracted = [extracted]
            plots = [p for p in extracted if isinstance(p, (Figure, ArtistAnimation))]

            if len(plots) != len(self.plot_names):
                raise ValueError("should be equal number of plots and names")

            print("Saving plots to the following files:")
            for name, plot in zip(self.plot_names, plots):
                plot_base = self.plots_dir / name
                match type(plot):
                    case mplf.Figure:
                        with open(plot_base.with_suffix(".pkl"), "wb") as pickle_file:
                            pickle.dump(plot, pickle_file)
                        cast(Figure, plot).savefig(plot_base.with_suffix(".png"), format='png')
                        print(f"\t\t{plot_base.with_suffix(".pkl")!s}")
                        print(f"\t\t{plot_base.with_suffix(".png")!s}")
                    case mpla.ArtistAnimation:
                        cast(ArtistAnimation, plot).save(filename=plot_base.with_suffix(".gif"), writer="pillow")
                        print(f"\t\t{plot_base.with_suffix(".gif")!s}")
            return func_ret

        return wrapper
