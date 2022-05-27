from __future__ import print_function

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import math
import copy

#### Task-space force control:

class ForceCtrl:
    def __init__(self, urbodx):
        self.robot = urbodx
        self.kp = np.array([100, 50, 50])
        self.kd = np.array([1, 0.1, 0.1]) * 0.1

        self.kpCartesian = np.array([2500, 2500, 2500])
        self.kdCartesian = np.array([40, 40, 40])

    def torque_cmd(self,leg_support,J_leg, F_leg,q_ref, q_mea, dq_ref, dq_mea,p_ref, p_mea, dp_ref, dp_mea,Gravity_comp):
        torque_cmd = np.array([0,0,0])
        if leg_support: ###stance leg
            torque_ff = self.stance_leg_forward(J_leg, F_leg)
        else: ###swing leg
            torque_ff = self.swing_leg_forward(J_leg, p_ref, p_mea, dp_ref, dp_mea)

        joint_track_torque = self.Joint_track_fb(q_ref, q_mea, dq_ref, dq_mea)
        Gra_comp = self.Gravity_comp(Gravity_comp)

        for i in range(0,3):
            torque_cmd[i] = torque_ff[i,0] + joint_track_torque[i] + Gra_comp[i]

        return torque_cmd



    # FR, FL, RR, RL
    def stance_leg_forward(self, J_leg, F_leg):
        torque_ff = -np.dot(J_leg.T, F_leg)
        # print(torque_ff)
        return torque_ff

    def swing_leg_forward(self, J_leg, p_ref, p_mea, dp_ref, dp_mea):
        F_statence = self.kpCartesian * (p_ref - p_mea) + self.kdCartesian * (dp_ref - dp_mea)
        torque_ff = -np.dot(J_leg.T, F_statence)
        return torque_ff

    def Joint_track_fb(self,q_ref, q_mea, dq_ref, dq_mea):
        torque_fb = self.kp * (q_ref - q_mea) + self.kd * (dq_ref - dq_mea)
        # print(torque_fb)
        return torque_fb

    def Gravity_comp(self, compensation):
        torque_gf = compensation
        # print(torque_gf)
        return torque_gf
