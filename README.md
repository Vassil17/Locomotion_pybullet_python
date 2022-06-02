# locomotion_pybullet_python

Build the pybullet simulation environment for legged robots motion planning and control(coding by python.3.6)

run "talos_nmpc_ik.py" for robot locomotion control

The Pinocchio dynamic library is used for robot-computation: 

Note: the .exe library (generated by external c++ code, see github repo: "Bipedal_locomotion_nmpc_for_pybullet") is employed for fast gait planing and optimization.

# Branches
(1) Main: bipedal walking: Talos robot test:

Reference: 

   1,Ding, J., Zhou, C., Xin, S., Xiao, X. and Tsagarakis, N.G., 2021. Nonlinear model predictive control for robust bipedal locomotion: exploring angular momentum and CoM height changes. Advanced Robotics, 35(18), pp.1079-1097.
   
   2,Ding, J., Xin, S., Lam, T.L. and Vijayakumar, S., 2021, May. Versatile locomotion by integrating ankle, hip, stepping, and height variation strategies. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 2957-2963). IEEE.

(2) Robot extend: Pinocchio test for floting-based legged robotics

Reference:

    3,Carpentier, J., Saurel, G., Buondonno, G., Mirabel, J., Lamiraux, F., Stasse, O. and Mansard, N., 2019, January. The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives. In 2019 IEEE/SICE International Symposium on System Integration (SII) (pp. 614-619). IEEE.


(3) Quadrupedal_locomotion: Torque-control framework for quadrupedal locomotion

Reference:

    4,Di Carlo, J., Wensing, P.M., Katz, B., Bledt, G. and Kim, S., 2018, October. Dynamic locomotion in the mit cheetah 3 through convex model-predictive control. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 1-9). IEEE.
    
    5,Focchi, M., Del Prete, A., Havoutis, I., Featherstone, R., Caldwell, D.G. and Semini, C., 2017. High-slope terrain locomotion for torque-controlled quadruped robots. Autonomous Robots, 41(1), pp.259-272.
   
