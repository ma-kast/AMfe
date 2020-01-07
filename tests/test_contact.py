

import unittest
from amfe.contact import JenkinsContact

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose, assert_raises


class BasicFunctionality(unittest.TestCase):

    def setUp(self):
        self.k_n = 30
        self. k_t = 20
        self.mu = 0.3
        return None

    def test_create_contact_object(self):
        contact = JenkinsContact()

    def test_initialize_attributes(self):
        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        self.assertAlmostEqual(self.k_n, contact.k_n)
        self.assertAlmostEqual(self.k_t, contact.k_t)
        self.assertAlmostEqual(self.mu, contact.mu)

        # Check if hysteresis states have been initialized
        nominal_initialization = np.zeros((2, 1))
        assert_array_equal(nominal_initialization, contact._f_t_previous)
        assert_array_equal(nominal_initialization, contact._u_rel_t_previous)


class ComputationOfKandf(unittest.TestCase):

    def setUp(self):
        self.k_n = 30
        self. k_t = 20
        self.mu = 0.3
        return None

    def test_separation(self):
        # A separated state
        u_rel = np.zeros((3, 1))
        u_rel[2] = 0.2
        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For separation, we expect no force and no stiffness contributions
        K_nominal = np.zeros((3, 3))
        f_nominal = np.zeros((3, 1))

        assert_array_equal(K, K_nominal)
        assert_array_equal(f_nominal, f)

    def test_separation_0(self):

        u_rel = np.zeros((3, 1))  # Separation
        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For separation, even u_rel 0, we expect no force and no stiffness contributions
        K_nominal = np.zeros((3, 3))
        f_nominal = np.zeros((3, 1))

        assert_array_equal(K, K_nominal)
        assert_array_equal(f_nominal, f)

    def test_only_contact(self):

        u_rel = np.zeros((3, 1))
        u_rel[2] = -0.2

        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For pure contact, we only expect fore contributions in the normal direction. However, the contributions in the
        # Jacobian are also in the tangential direction, even for zero relative displacement
        K_nominal = np.zeros((3, 3))
        K_nominal[2, 2] = self.k_n
        K_nominal[0, 0] = self.k_t
        K_nominal[1, 1] = self.k_t
        f_nominal = np.zeros((3, 1))
        f_nominal[2] = u_rel[2] * self.k_n

        assert_array_equal(K, K_nominal)
        assert_array_equal(f_nominal, f)

    def test_sticking_contact_1D(self):

        u_rel = np.zeros((3, 1))
        u_rel[0] = -.02
        u_rel[2] = -0.2

        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For sticking contact in 1D, we only expect force contributions in the normal and given tangential direction.
        K_nominal = np.zeros((3, 3))
        K_nominal[2, 2] = self.k_n
        K_nominal[0, 0] = self.k_t
        K_nominal[1, 1] = self.k_t
        f_nominal = np.zeros((3, 1))
        f_nominal[2] = u_rel[2] * self.k_n
        f_nominal[0] = u_rel[0] * self.k_t

        assert_array_equal(K, K_nominal)
        assert_array_equal(f_nominal, f)


    def test_slipping_contact_1D(self):

        u_rel = np.zeros((3, 1))
        u_rel[0] = -.3
        u_rel[2] = -0.1

        # Check that we actually choose the parameters in the slipping regime
        f_limit = np.abs(u_rel[2]*self.k_n* self.mu)
        f_trial = u_rel[0] * self.k_t
        self.assertGreater(np.abs(f_trial), f_limit)


        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For slipping contact in 1D, we only expect force contributions in the normal and given tangential direction.
        # However, the Jacobian will have entries in the other tangential direction (Rotation property.)
        f_nominal = np.zeros((3, 1))
        f_nominal[2] = u_rel[2] * self.k_n
        f_nominal[0] = f_limit * (-1) # The -1 is the direction of the tangential displacement


        K_nominal = np.zeros((3, 3))
        K_nominal[2, 2] = self.k_n
        K_nominal[1, 1] = self.k_t * f_limit / abs(f_trial)
        K_nominal[0, 2] = self.k_n * self.mu * (-1) # The -1 is the direction of the tangential displacement


        assert_array_equal(K, K_nominal)
        assert_array_equal(f_nominal, f)

    def test_slipping_contact_2D(self):

        u_rel = np.zeros((3, 1))
        u_rel[0] = -.03
        u_rel[1] = -.04
        u_rel[2] = -0.1

        # Check that we actually choose the parameters in the slipping regime
        f_limit = np.abs(u_rel[2]*self.k_n* self.mu)
        f_trial = u_rel[:2] * self.k_t
        f_norm = np.linalg.norm(f_trial)
        direction = f_trial/f_norm
        self.assertGreater(f_norm, f_limit)




        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For slipping contact in 2D, we expect force contributions in all directions.

        f_nominal = np.zeros((3, 1))
        f_nominal[2] = u_rel[2] * self.k_n
        f_nominal[:2] = f_limit * direction

        K_nominal = np.zeros((3, 3))
        K_nominal[2, 2] = self.k_n
        K_nominal[:2, :2] = self.k_t* f_limit / f_norm**3 * np.array([[f_trial[1] ** 2, f_trial[0] * f_trial[1]],
                                          [f_trial[0] * f_trial[1], f_trial[0] ** 2]]).reshape(2,2)
        K_nominal[:2, 2:] = self.k_n*self.mu * direction

        assert_array_almost_equal(K, K_nominal)
        assert_array_almost_equal(f_nominal, f)


    def test_sticking_contact_2D(self):

        u_rel = np.zeros((3, 1))
        u_rel[0] = -.02
        u_rel[1] = -0.03
        u_rel[2] = -0.2

        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        K, f = contact.K_and_f(u_rel)

        # For sticking contact in 1D, we only expect force contributions in the normal and given tangential direction.
        K_nominal = np.zeros((3, 3))
        K_nominal[2, 2] = self.k_n
        K_nominal[0, 0] = self.k_t
        K_nominal[1, 1] = self.k_t
        f_nominal = np.zeros((3, 1))
        f_nominal[2] = u_rel[2] * self.k_n
        f_nominal[:2] = u_rel[:2] * self.k_t

        assert_array_almost_equal(K, K_nominal)
        assert_array_almost_equal(f_nominal, f)


class PropagationOfStates(unittest.TestCase):

    def setUp(self):
        self.k_n = 30
        self. k_t = 20
        self.mu = 0.3
        return None

    def test_states_update_always(self):
        # Start in contact
        u_rel = u_rel = np.zeros((3, 1))
        u_rel[0] = -.02
        u_rel[1] = -0.03
        u_rel[2] = -0.2
        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)

        _, f = contact.K_and_f(u_rel)
        assert_allclose(contact._u_rel_t, u_rel[:2])
        assert_allclose(contact._f_t, f[:2])

        # Separate
        u_rel[1] = -0.01
        u_rel[2] = 0.2
        _, f = contact.K_and_f(u_rel)
        assert_allclose(contact._u_rel_t, u_rel[:2])
        assert_allclose(contact._f_t, f[:2])

        # Separation
        u_rel[1] = -0.1
        _, f = contact.K_and_f(u_rel)
        assert_allclose(contact._u_rel_t, u_rel[:2])
        assert_allclose(contact._f_t, f[:2])

        # Contact again
        u_rel[1] = 0.1
        u_rel[2] = -0.2
        _, f = contact.K_and_f(u_rel)
        assert_allclose(contact._u_rel_t, u_rel[:2])
        assert_allclose(contact._f_t, f[:2])

    def test_states_propagate_on_call(self):
        # Start in contact
        u_rel = np.zeros((3, 1))
        u_rel[0] = -.02
        u_rel[1] = -0.03
        u_rel[2] = -0.2
        contact = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)

        _, f = contact.K_and_f(u_rel)
        # Check that we are actually testing the call
        assert_raises(AssertionError, assert_allclose, contact._f_t, contact._f_t_previous)
        assert_raises(AssertionError, assert_allclose, contact._u_rel_t, contact._u_rel_t_previous)

        contact._update_states()

        # Compare after call
        assert_allclose(contact._f_t, contact._f_t_previous)
        assert_allclose(contact._u_rel_t, contact._u_rel_t_previous)


    def test_states_propagate_in_tuple(self):
        # This tests more python behavior than the actual implementation. But it gives us
        # confidence that things work as expected.
        contact_1 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contact_2 = JenkinsContact(k_n=self.k_n, k_t=self.k_t, mu=self.mu, u_rel_t=None, f_t=None)
        contacts = (contact_1, contact_2)

        u_rel = np.zeros((3, 1))
        u_rel[0] = -.02
        u_rel[1] = -0.03
        u_rel[2] = -0.2

        contact = contacts[0]
        _, _ = contact.K_and_f(u_rel)
        assert_allclose(contact_1._f_t, contact._f_t)
        assert_allclose(contacts[0]._f_t, contact._f_t)
        contact._update_states()
        assert_allclose(contact_1._f_t_previous, contact._f_t_previous)
        assert_allclose(contacts[0]._f_t_previous, contact._f_t)

if __name__ == '__main__':
    unittest.main()
