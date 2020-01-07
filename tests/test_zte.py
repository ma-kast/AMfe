

import unittest
from amfe.contact import JenkinsContact
from amfe.element.quad4ZTE import Quad4ZTE

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose, assert_raises


class BasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.k_n = 30
        self. k_t = 20
        self.mu = 0.3
        contact1 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact2 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact3 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact4 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        self.contacts = (contact1, contact2, contact3, contact4)
        return None

    def test_create_zte_element(self):
        element = Quad4ZTE(self.contacts)



class CorrectComputations(unittest.TestCase):

    def setUp(self):
        self.k_n = 30
        self. k_t = 20
        self.mu = 0.3
        contact1 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact2 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact3 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact4 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        self.contacts = (contact1, contact2, contact3, contact4)
        return None

    def test_compute_K_and_f(self):
        X = np.zeros((8,3))
        # X can have any shape in the y-z plane
        X[1, 1] = 0.8
        X[1, 2] = 0.2
        X[2, 1] = 1
        X[2, 2] = 1.1
        X[3, 2] = 1
        X[4:, :] = X[:4, :] # I am not actually sure how the other quad should be ordered. It might be the other way
        # round?

        # The displacement can be any combination for sticking
        u_n = -0.001
        u_t1 = -0.0001
        u_t2 = 0.0002

        # Compute the area because of Gauss integration
        vec_1 = X[1, :] - X[0, :]
        vec_2 = X[3, :] - X[0, :]
        vec_3 = X[1, :] - X[2, :]
        vec_4 = X[3, :] - X[2, :]
        area = 0.5 * np.linalg.norm(np.cross(vec_1, vec_2), ord=2) + 0.5 * np.linalg.norm(np.cross(vec_3, vec_4), ord=2)

        u = X * 0

        # All nodal dofs have the same displacement. Thus, the exact shape of the quad in the plane, should
        #  not influence the result. (Apart from scaling with the area)
        u[:4, 0] = u_n
        u[:4, 1] = u_t1
        u[:4, 2] = u_t2

        X = X.flatten()
        u = u.flatten()

        # We can thus compute the nominal forces easily
        f_nominal = np.zeros((3, 1))
        f_nominal[0] = u_n * self.k_n
        f_nominal[1] = u_t1 * self.k_t
        f_nominal[2] = u_t2 * self.k_t
        f_nominal = f_nominal * area

        # Check the result of the quad
        element = Quad4ZTE(self.contacts)
        K, f = element.k_and_f_int(X, u)
        f_mat = f.reshape(8,3)
        # Compute sum of all dofs for comparison
        f_total = np.sum(f_mat[:4, :], axis=0)

        assert_array_almost_equal(f_total.flatten(), f_nominal.flatten())