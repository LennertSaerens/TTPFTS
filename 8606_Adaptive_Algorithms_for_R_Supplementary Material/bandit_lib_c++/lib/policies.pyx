# distutils: language = c++
# coding=utf-8
cimport cython
import numpy as np
from .bandits import Bandit
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

cdef class Policy(object):
    def __init__(self, Bandit py_bandit):
        self.K = py_bandit.K
        self.D = py_bandit.D
        self.sigma = py_bandit.sigma
        self.dim = py_bandit.D
        self.action_space = py_bandit.action_space
        self.py_bandit = py_bandit

cdef class p_auer(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_auer(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1, double eps=0., size_t m = 1<<16):
        return self.policy_ref.loop(seed, delta, eps, m)

cdef class p_ape(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_ape(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1, double eps_1=0., double eps_2=0., size_t m = 1<<16):
        return self.policy_ref.loop(seed, delta, eps_1, eps_2, m)


cdef class p_unif(Policy):
    def __init__(self, Bandit py_bandit):
        super().__init__(py_bandit)
    def __cinit__(self, Bandit py_bandit):
        self.bandit_ref = py_bandit.bandit_ref
        self.policy_ref = new psi_uniform(deref(self.bandit_ref))
    def loop(self, size_t seed=42, double delta=0.1, double eps_1=0., double eps_2=0., size_t m = 2**16):
        return self.policy_ref.loop(seed, delta, eps_1, eps_2, m)