"""Analysis Module."""
import matplotlib.pyplot as plt

from thermosnooker._utils.decorators import SaveOutput


def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    return


@SaveOutput("task10")
def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.

    Returns:
        Figure: The MultiBallSimulation simulation plot
    """
    return


@SaveOutput(["task11a", "task11b"])
def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Firgure]: The histograms (distance from centre, inter-ball spacing).
    """
    return


@SaveOutput(["task12a", "task12b", "task12c", "task12d"])
def task12():
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy and momentum are conserved.
    Additionally we shall investigate the pressure evolution of the system. Ensure that the 4 figures
    outlined in the project script are returned.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, momentum_x, momentum_y ratios
                                               as well as pressure evolution.
    """
    return


@SaveOutput(["task13a", "task13b", "task13c"])
def task13():
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    return


@SaveOutput("task14")
def task14():
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio defined in
    the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    return


@SaveOutput("task15")
def task15():
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    return


@SaveOutput(["task16a", "task16b"])
def task16():
    """
    Task 16.

    In this function we shall also be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio
    and volume fraction defined in the project brief. We shall fit this temperature ratio before
    plotting the VDW b parameters radii dependence.

    Returns:
        tuple[Figure, Figure]: The ratio figure and b parameter figure.
    """
    return


@SaveOutput("task17")
def task17():
    """
    Task 17.

    In this function we shall run a Brownian motion simulation and plot the resulting trajectory of the 'big' ball.

    Returns:
        Figure: The Brownian motion simulation plot.
    """
    return


@SaveOutput("task18")
def task18():
    """
    Task 18.

    In this function we shall calculate and plot the radial dependence of the mean free path and compare to the
    dilute-gas Boltzmann mean free path. We shall then investigate the Enskog correction as the larger radii put us in
    dense-gas region.

    Returns:
        Figure: The plot of your mean free path investigation.
    """
    return


@SaveOutput(["task19a", "task19b", "task19c"])
def task19():
    """
    Task 19.

    In this function, we shall be computing the radial distribution function. We will see what the function looks like
    at time t = 0 as well as a later time t for a MultiBallSimulation where only a single ball has some velocity.
    We will also create an animation to so the evolution of this function as our simulation progresses.

    Returns:
        tuple[Figure, Figure, ArtistAnimation]: The g(r) histograms for t = 0, t = some time later, g(r) animation
    """
    return


if __name__ == "__main__":

    # Run task 9 function
    BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    # FIG10 = task10()

    # Run task 11 function
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    # FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    # FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    # FIG14 = task14()

    # Run task 15 function
    # FIG15 = task15()

    # Run task 16 function
    # FIG16_RATIO, FIG16_BPARAM = task16()

    # Run task 17 function
    # FIG17 = task17()

    # Run task 18 function
    # FIG18 = task18()

    # Run task 19 function
    # FIG19_HIST1, FIG19_HIST2, FIG19_ANIM = task19()

    plt.show()
