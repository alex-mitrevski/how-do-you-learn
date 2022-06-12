Proximal Policy Optimization (PPO)
==================================

The implementation of PPO generally follows the original description of the algorithm in::

    J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov,
    "Proximal Policy Optimization Algorithms", CoRR, vol. abs/1707.06347, 2017.


but does not use a baseline value, i.e. I currently don't estimate the value function in my implementation.
