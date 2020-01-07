# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module for contact handling withing the FE context.
Up to now, only contact that does not require additional dofs is implemented. The contact model under consideration uses
a return mapping algorithm, i.e. it requires the previous state. For now, this info is saved in the contact model.
The contact model should be able to return the forces and contributions to the Jacobian.
For now, contact depends only on the displacements.
Different implementations might use velocity and will have contributions that are not in the stiffness matrix K.

"""

import numpy as np

__all__ = ['Contact',
           'JenkinsContact'
           ]

use_fortran = False


# use_fortran = False


class Contact:
    def __init__(self):
        self._observers = list()

    def add_observer(self, observer, verbose=True):
        self._observers.append(observer)
        if verbose:
            print('Added observer to contact')

    def remove_observer(self, observer, verbose=True):
        self._observers.remove(observer)
        if verbose:
            print('Removed observer from contact')

    def notify(self):
        for observer in self._observers:
            observer.update()


class HysteresisContact(Contact):
    """
    Base class for contacts that require the previous state and only need the displacements for computation.
    Follows the implementation ideas of Afzal 2016.
    """

    def __init__(self):
        super().__init__()

    def K_and_f(self, u_rel, Minv):
        """
        Compute Jacobian contribution/stiffness and nonlinear forces due to the contact state

        Parameters
        ----------
        u_rel : ndarray
            relative displacements at point of contact, shape (3,)

        Returns
        -------
        K : ndarray
            Jacobian contribution of the contact state,
            shape: (3,3)
        f : ndarray
            Contact forces, shape: (3,)
        """
        pass


class JenkinsContact(HysteresisContact):
    r"""
    2D Jenkins element that models contact in a plane.

    The contact is detected along the normal direction (z) and contact forces are computed with a
    linear penalty approach.
    The friction force is computed in the plane with the Masing-Hypothesis, Jenkins formulation of the Hysteresis.

    see Afzal 2016  for a derivation of this  contact model.

    """

    def __init__(self, k_n=30, k_t=30, mu=0.3, u_rel_t=None, f_t=None):
        """

                Parameters
                ----------
                k_n : float
                    Penalty stiffness in normal direction
                k_t : float
                    Stiction/Spring stiffness in tangential direction.
                    TODO: Potentially shape (2,) for non-isotropic friction
                mu : float
                    Friction coefficient of the Coulomb limit

                u_rel_t : shape(2,1) if given, else None
                    Initial relative displacement, should be 0 apart from exceptional cases. Needs to be set with f_t

                f_t : shape(2,1) if given, else None
                    Initial tangential force, should be 0 apart from exceptional cases. Needs to be set with u_rel_t
                Returns
                -------
                None
                """

        super().__init__()

        self.k_n = k_n
        self.k_t = k_t
        self.mu = mu

        if u_rel_t is None or f_t is None:
            self._u_rel_t = np.zeros((2, 1))
            self._f_t = np.zeros((2, 1))
        else:
            self._u_rel_t = u_rel_t
            self._f_t = f_t

        self._update_states()

    def __repr__(self):
        """
        repr(obj) function for smart representing for debugging
        """
        return 'amfe.contact.JenkinsContact(%f,%f)' \
               % (self.k_n, self.mu)
        # TODO: Write a nice presentation for the Jenkins element

    def _update_states(self):
        """"
        Update internal states. This is required because the Jenkins element needs a memory of its previous state.
        """
        self._u_rel_t_previous = self._u_rel_t
        self._f_t_previous = self._f_t

    def K_and_f(self, u_rel, Minv = None):
        if Minv is None:
            Minv = np.eye(2)

        K = np.zeros((3, 3))
        f = np.zeros((3, 1))

        # Always update the relative displacement
        self._u_rel_t = u_rel[:2]

        # Check if we are actually in contact by looking at the normal dof.
        if u_rel[2] < 0:
            # We are in contact
            f[2] = self.k_n * u_rel[2]
            K[2, 2] = self.k_n

            # Compute trial stress
            trial_stress = self._f_t_previous + self.k_t * (self._u_rel_t - self._u_rel_t_previous)

            # Check if we are in the sticking or slipping regime

            f_limit = np.abs(f[2] * self.mu)  # Coulomb friction limit
            f_trial_corr = np.sqrt(trial_stress.T  @Minv @ trial_stress)  # norm of trial stress in correct metric?
            f_trial = f_trial_corr #np.linalg.norm(trial_stress, ord =2)
            if f_trial_corr <= f_limit:
                # Sticking regime
                self._f_t = trial_stress
                f[:2] = self._f_t
                K[0, 0] = self.k_t
                K[1, 1] = self.k_t

            else:
                # Slipping regime
                # Be careful not to divide by zero
                if f_trial > 0:
                    self._f_t = trial_stress / f_trial * f_limit
                    # Derivative with respect to normal displacement
                    K[:2, 2:] = trial_stress / f_trial * self.k_n * self.mu
                    # Derivative with respect to tangential displacements
                    diag_part = np.eye(2) / f_trial
                    skewed_part = np.outer(trial_stress, trial_stress) @ Minv /f_trial**3
                    K[:2, :2] = self.k_t * f_limit *  (diag_part -skewed_part)
                    f[:2] = self._f_t
                else:
                    self._f_t = np.zeros((2, 1))  # We should never reach this statement
                    print("Warning, this statement should not be reachable. Slipping regime with 0 trial stress?!")

        else:
            # We are not in contact. Update tangential stresses to be 0.
            self._f_t = f[:2]

        return K, f
