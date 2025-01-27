from abc import ABC, abstractmethod
from math import e, log, sqrt


class RadialBasisFunc(ABC):
    """
    Interface for radial basis functions.
    """
    @abstractmethod
    def f(self, d :float) -> float:
        """
        Computes the value of the radial basis function.

        Parameters
        ----------
        d : float
            Distance from the center of the radial basis function.

        Returns
        -------
        : float
            The value of the radial basis function.
        """
        pass



class GaussianRBF(RadialBasisFunc):
    """
    Gaussian radial basis function.
    """
    def compute_sigma(p :float, d :float) -> float:
        """
        Inverse of the Gaussian radial basis function.
        Given the distance ``d`` from the centre and the desired output ``p``, 
        it computes the value of ``sigma`` consequently.

        Parameters
        ----------
        p : float
            Desired output of the Gaussian radial basis function.
        d : float
            Distance from the center of the radial basis function.

        Returns
        -------
        : float
            Value of the shape parameter to get in output ``p`` given as input the distance ``d``.
        """
        return sqrt(-log(1-p)) / d


    def __init__(self, sigma :float):
        """
        Parameters
        ----------
        sigma : float
            Shape parameter of the Gaussian radial basis function.
        """
        super().__init__()
        self.sigma = sigma
        return
    
    
    def f(self, d :float) -> float:
        return 1 - e **(-(d * self.sigma)**2)