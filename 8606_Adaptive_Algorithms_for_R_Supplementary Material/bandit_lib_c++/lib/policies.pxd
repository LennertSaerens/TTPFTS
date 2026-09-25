# distutils: language = c++
# coding=utf-8
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from .bandits cimport bandit, Bandit
# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef extern from "../src/cpp/policies.cxx":
    pass
# Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass policy:
        policy() except+;
        policy(bandit& bandit_ref) except+;
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        pair[pair[size_t, np.npy_bool], vector[size_t]] loop() nogil;

# Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass psi_auer(policy):
        size_t K;
        size_t dim;
        size_t D;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta;
        double eps;
        psi_auer() except+;
        psi_auer(bandit &) except+;
        pair[pair[np.npy_bool, vector[size_t]], vector[size_t]] loop(const size_t&, const double&, const double&, const size_t&) nogil;

#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass psi_ape(policy):
        size_t K, D;
        size_t dim;
        size_t m;
        double sigma;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta;
        double eps_1, eps_2;
        psi_ape() except+;
        psi_ape(bandit &) except+;
        pair[pair[np.npy_bool, vector[size_t]], vector[size_t]] loop(const size_t&, const double&, const double&, const double&, const size_t&) nogil;

#Declare the class with cdef
cdef extern from "../src/cpp/policies.hpp":
    cdef cppclass psi_uniform(policy):
        size_t K, D, dim, m;
        vector[size_t] action_space;
        bandit* bandit_ref;
        double delta, eps_1, eps_2, sigma;
        psi_uniform() except+;
        psi_uniform(bandit &) except+;
        pair[pair[size_t, np.npy_bool], vector[size_t]] loop(const size_t&, const double&, const double&, const double&, const size_t&) nogil;
# Define Python interfaces
cdef class Policy:
    cdef readonly size_t K;
    cdef readonly size_t D;
    cdef readonly size_t dim;
    cdef readonly double sigma;
    cdef Bandit py_bandit;
    cdef bandit* bandit_ref
    #cdef policy* policy_ref;
    cdef readonly vector[size_t] action_space;

cdef class p_auer(Policy):
    cdef psi_auer* policy_ref
cdef class p_ape(Policy):
    cdef psi_ape* policy_ref
cdef class p_unif(Policy):
    cdef psi_uniform* policy_ref