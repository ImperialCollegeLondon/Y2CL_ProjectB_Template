import os
from functools import wraps
import pickle
import matplotlib.testing.decorators as mpltd
import matplotlib.pyplot as plt
import matplotlib as mpl

def check_figures_equal(*, ref_path=None, tol=0):
    """
    Decorator for test cases that generate and compare two figures.

    The decorated function must take two keyword arguments, *fig_test*
    and *fig_ref*, and draw the test and reference images on them.
    After the function returns, the figures are saved and compared.

    This decorator should be preferred over `image_comparison` when possible in
    order to keep the size of the test suite from ballooning.

    Parameters
    ----------
    tol : float
        The RMS threshold above which the test is considered failed.

    Raises
    ------
    RuntimeError
        If any new figures are created (and not subsequently closed) inside
        the test function.

    Examples
    --------
    Check that calling `.Axes.plot` with a single argument plots it against
    ``[0, 1, 2, ...]``::

        @check_figures_equal()
        def test_plot(fig_test, fig_ref):
            fig_test.subplots().plot([1, 3, 5])
            fig_ref.subplots().plot([0, 1, 2], [1, 3, 5])

    """
    ALLOWED_CHARS = set(mpltd.string.digits + mpltd.string.ascii_letters + '_-[]()')
    KEYWORD_ONLY = mpltd.inspect.Parameter.KEYWORD_ONLY
    if not isinstance(ref_path, str) or not ref_path:
        raise ValueError("need a reference path")

    def decorator(func):

        _, result_dir = mpltd._image_directories(func)
        old_sig = mpltd.inspect.signature(func)

        @wraps(func)
        def wrapper(*args, request, **kwargs):
            if 'request' in old_sig.parameters:
                kwargs['request'] = request

            file_name = "".join(c for c in request.node.name
                                if c in ALLOWED_CHARS)
            try:

                ref_file_path = os.path.join(os.path.dirname(__file__),
                                             "plots",
                                             ref_path)
                with open(ref_file_path, "rb") as ref_file:
                    fig_ref = pickle.load(ref_file)
                    if not isinstance(fig_ref, mpl.figure.Figure):
                        fig_ref = fig_ref.get_figure()

                #with mpltd._collect_new_figures() as figs:
                fig_test = func(*args, **kwargs)
                if not isinstance(fig_test, mpl.figure.Figure):
                    fig_test = fig_test.get_figure()
                #if figs:
                #    raise RuntimeError('Number of open figures changed during '
                #                       'test. Make sure you are plotting to '
                #                       'fig_test or fig_ref, or if this is '
                #                       'deliberate explicitly close the '
                #                       'new figure(s) inside the test.')
                test_image_path = result_dir / (file_name + ".png")# + ext)
                ref_image_path = result_dir / (file_name + "-expected.png")# + ext)
                fig_test.savefig(test_image_path)
                fig_ref.savefig(ref_image_path)
                mpltd._raise_on_image_difference(
                    ref_image_path, test_image_path, tol=tol
                )
            finally:
                plt.close(fig_test)
                plt.close(fig_ref)

        parameters = [
            param
            for param in old_sig.parameters.values()
            if param.name not in {"fig_test"}
        ]
        if 'request' not in old_sig.parameters:
            parameters += [mpltd.inspect.Parameter("request", KEYWORD_ONLY)]
        new_sig = old_sig.replace(parameters=parameters)
        wrapper.__signature__ = new_sig

        return wrapper

    return decorator