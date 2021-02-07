from __future__ import print_function

import os
from os.path import dirname, join, abspath
import sys

import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *

import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
from pino_robot_ik import CLIK


import pybullet
import pybullet_data
from sim_env import SimEnv
from sim_robot import SimRobot
import time
import math
import datetime


global base_homing
global pr_homing
global pl_homing

########whold-body simu
Full_body_simu = True
##########setting floating base for pinocchio
Freebase = True



def addFreeFlyerJointLimits(robot):
    rmodel = robot.model

    ub = rmodel.upperPositionLimit
    ub[:7] = 1e-6
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1e-6
    rmodel.lowerPositionLimit = lb


mesh_dir = '/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos'
# You should change here to set up your own URDF file
if Full_body_simu:
    urdf_filename = mesh_dir + '/talos_description/urdf/talos_full_no_grippers.urdf'
else:
    urdf_filename = mesh_dir + '/talos_description/urdf/talos_lower_body_mesh_updated.urdf'

### pinocchio load urdf
if Freebase:
    robot =  RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir, pin.JointModelFreeFlyer())
    addFreeFlyerJointLimits(robot)
else:
    robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir)
### explore the model class
# for name, function in robot.model.__class__.__dict__.items():
#     print(' **** %s: %s' % (name, function.__doc__))
# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
# print('standard model: dim=' + str(len(robot.model.joints)))
# for jn in robot.model.joints:
#     print(jn)
# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' )

# find lower-leg joint idx in pinocchio
joints_desired_l = ['leg_left_1_joint','leg_left_2_joint','leg_left_3_joint', 'leg_left_4_joint', 'leg_left_5_joint','leg_left_6_joint']
joints_desired_r = ['leg_right_1_joint','leg_right_2_joint','leg_right_3_joint', 'leg_right_4_joint', 'leg_right_5_joint','leg_right_6_joint']
idr =[]
idl =[]
for i in range(0,len(joints_desired_r)):
    idl.append(robot.model.getJointId(joints_desired_l[i]))
    idr.append(robot.model.getJointId(joints_desired_r[i]))
print("id_r",idr)
print("id_l",idl)

##### reset base pos and orn,  only workable in floating-based model
# robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array([1,0.5,0.8]))

# ##### right, lef leg end joints
# q = robot.q0
# pin.forwardKinematics(robot.model,robot.data,q)
# pr = robot.data.oMi[idr[5]].translation
# pl = robot.data.oMi[idl[5]].translation
# print("right ankle joint pos:", pr)
# print("left ankle joint pos:", pl)
# print(" base_position:",robot.model.jointPlacements[1])
#
# robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array([1,0,0.8))
# ##### right, lef leg end joints
# q = robot.q0
# pin.forwardKinematics(robot.model,robot.data,q)
# pr = robot.data.oMi[idr[5]].translation
# pl = robot.data.oMi[idl[5]].translation
# print("right ankle joint pos:", pr)
# print("left ankle joint pos:", pl)
# print(" base_position_update:",robot.model.jointPlacements[1])
############################### pybullet load #####################################################


### intial pose for talos in pybullet
sim_rate = 100
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)

# urobtx = SimRobot(urdfFileName=urdf_filename,
#                  basePosition=[0, 0, 1.1],
#                  baseRPY=[0, 0, 0],
#                  jointPositions=Homing_pose)
urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=[0, 0, 1.1],
                 baseRPY=[0, 0, 0])
urobt = urobtx.id

num_joint = urobtx.getNumActuatedJoints()
if Full_body_simu:
    Homing_pose = np.zeros(num_joint)
    ### upper arm
    Homing_pose[4] =  -0.2
    Homing_pose[5] =  0.1
    Homing_pose[7] =  -0.6
    Homing_pose[11] =  0.2
    Homing_pose[12] =  -0.1
    Homing_pose[14] =  -0.6
    ## lower leg
    Homing_pose[-10] =  -0.2
    Homing_pose[-9] =  0.4
    Homing_pose[-8] =  -0.2
    Homing_pose[-4] =  -0.2
    Homing_pose[-3] =  0.4
    Homing_pose[-2] =  -0.2
else:
    ## lower leg
    Homing_pose = np.zeros(num_joint)
    Homing_pose[2] =  -0.2
    Homing_pose[3] =  0.4
    Homing_pose[4] =  -0.2
    Homing_pose[8] =  -0.2
    Homing_pose[9] =  0.4
    Homing_pose[10] =  -0.2

print("Homing_pose:",Homing_pose)

t_homing = 1






useRealTimeSimulation = 0

t=0.
i=0
pr_homing_fix = []
pl_homing_fix = []
base_home_fix = []
while i<10000:
    #################ste base pos and orn in bullet
    # pybullet.resetBasePositionAndOrientation(urobt,[0,0,1.1],[0,0,0,1])
    # input()
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        t = t + dt
    ############## robot control loop##################################################
    if t<t_homing:            ############# initial pose
        Homing_pose_t = Homing_pose*math.sin(t/t_homing/2.*math.pi)
        q = robot.q0
        if Freebase: ### note the in pinocchio, freebase has seven DoF
            if Full_body_simu:
                q[0+7:6+7] = Homing_pose_t[-12:-6]
                q[6+7:12+7] = Homing_pose_t[-6:]
            else:
                q[0+7:12+7] = Homing_pose_t
        else:
            # q = robot.q0
            if Full_body_simu:
                q[0:6] = Homing_pose_t[-12:-6]
                q[6:12] = Homing_pose_t[-6:]
            else:
                q  = Homing_pose_t*math.sin(t/t_homing/2.*math.pi)


        urobtx.setActuatedJointPositions(Homing_pose_t)
        lsx_ori = pybullet.getBasePositionAndOrientation(urobt)[0]
        # left_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=6)[4]
        # right_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=13)[4]
        # base_homing = np.array(lsx_ori)
        if Freebase:
            robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array(lsx_ori))
        pin.forwardKinematics(robot.model,robot.data,q)
        pr_homing = robot.data.oMi[idr[5]].translation
        pl_homing = robot.data.oMi[idl[5]].translation
        base_home_fix = lsx_ori
        pl_homing_fix = tuple(pl_homing)
        pr_homing_fix = tuple(pr_homing)
        print("pl_homing_fix",pl_homing_fix)
    else:
        ########### set endeffector id for ik using pinocchio
        JOINT_IDl = idl[5]
        JOINT_IDr = idr[5]

        ######## set reference trajectory
        if Freebase: #routine1: change the base position and orientation for pinocchio IK: time-cost becauseof the redundant freedom
            des_base = np.array([0,
                               0.05 * (math.sin((t - t_homing) * 50 * math.pi / 180)),
                               -0.05 * abs(math.sin((t - t_homing) * 50 * math.pi / 180))]) + np.array(base_home_fix)
            robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
            des_pl = np.array(pl_homing_fix)
            oMdesl = pin.SE3(np.eye(3), des_pl)
            des_pr = np.array(pr_homing_fix)
            oMdesr = pin.SE3(np.eye(3), des_pr)
            # print("base_homing_set", robot.model.jointPlacements[1].translation)
        else:  ####routine2: set the based position in local framework(note that always zeros), transform the base function in the
            des_pl = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pl_homing_fix)
            oMdesl = pin.SE3(np.eye(3), des_pl)
            des_pr = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pr_homing_fix)
            oMdesr = pin.SE3(np.eye(3), des_pr)

        ############ Jacobian-based IK
        IK_left_leg = CLIK(robot,oMdesl, JOINT_IDl,Freebase)
        IK_right_leg = CLIK(robot,oMdesr, JOINT_IDr,Freebase)

        q = robot.q0
        if t<t_homing+0.05:
            if Freebase:
                q[7:7+num_joint]  = Homing_pose
            else:
                q  = Homing_pose

        ql, Jl = IK_left_leg.ik_Jacobian(q=q, Freebase=Freebase,eps=1e-5, IT_MAX=1000, DT=1e-1, damp=1e-6)
        qr, Jr = IK_right_leg.ik_Jacobian(q=q, Freebase=Freebase,eps=1e-5, IT_MAX=1000, DT=1e-1, damp=1e-6)

        q_ik = Homing_pose
        if Freebase:
            q_ik[-12:-6] = ql[0+7:6+7]
            q_ik[-6:] = qr[6+7:12+7]
        else:
            q_ik[-12:-6] = ql[0:6]
            q_ik[-6:] = qr[6:12]
        # print("q_ik:", q_ik)

        ######## joint command###########################
        urobtx.setActuatedJointPositions(q_ik)
    # sim_env.step()
    i+=1
    pybullet.stepSimulation()

