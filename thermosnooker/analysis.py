"""Analysis Module."""
import matplotlib.pyplot as plt


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


def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't seeing errors like balls sticking
    together or escaping the container.
    """


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


def task15():
    """
    Task 15.

    In this function we shall also be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio
    and volume fraction defined in the project brief. We shall fit this temperature ratio before
    plotting the VDW b parameters radii dependence.

    Returns:
        tuple[Figure, Figure]: The ratio figure. and b parameter figure.
    """
    return


def task16():
    """
    Task 16.

    In this function we shall plot a histogram to investigate how the speeds of the balls evolve from the initial
    value. We shall then compare this to the Maxwell-Boltzmann distribution. Ensure that this function returns
    the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    return


def task17():
    """
    Task 17.

    In this function we shall run a Brownian motion simulation and plot the resulting trajectory of the 'big' ball.

    Returns:
        Figure: The track plot showing the motion of the 'big' ball
    """
    return


if __name__ == "__main__":

    # Run task 9 function
    BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    # task10()

    # Run task 11 function
    # FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    # FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    # FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    # FIG14 = task14()

    # Run task 15 function
    # FIG15_RATIO, FIG15_BPARAM = task15()

    # Run task 16 function
    # FIG16 = task16()

    # Run task 17 function
    # FIG17 = task17()

    plt.show()
