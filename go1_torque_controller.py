#### python envir setup
from __future__ import print_function

import copy
import os
from os.path import dirname, join, abspath
import sys

from pathlib import Path


### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                        #### IK solver
from robot_tracking_controller import Gait_Controller #### controller
from LIP_motion_planner import Gait                   #### Gait planner

##### numpy
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

##### pybullet
import pybullet
import pybullet_data
from sim_env import SimEnv
from sim_robot import SimRobot
import time
import math
import datetime

##### subprocess for external .exe
import subprocess

import scipy

################################ pinocchio urdf setup ##################################
def addFreeFlyerJointLimits(robot):
    rmodel = robot.model

    ub = rmodel.upperPositionLimit
    ub[:7] = 1e-6
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1e-6
    rmodel.lowerPositionLimit = lb

############################ NMPC c++ run for gait generation ##########################################################
#convert string  to number
def str2num(LineString, comment='#'):
    from io import StringIO as StringIO
    import re, numpy

    NumArray = numpy.empty([0], numpy.int16)
    NumStr = LineString.strip()
    # ~ ignore comment string
    for cmt in comment:
        CmtRe = cmt + '.*$'
        NumStr = re.sub(CmtRe, " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    # ~ delete all non-number characters,replaced by blankspace.
    NumStr = re.sub('[^0-9.e+-]', " ", NumStr, count=0, flags=re.IGNORECASE)

    # ~ Remove incorrect combining-characters for double type.
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    if len(NumStr.strip()) > 0:
        StrIOds = StringIO(NumStr.strip())
        NumArray = numpy.genfromtxt(StrIOds)

    return NumArray

def run_nmpc_external_ext(j,cpptest):
    b = str(j)
    # if os.path.exists(cpptest):
    rc, out = subprocess.getstatusoutput(cpptest + ' ' + b)
    donser = str2num(out)

    return donser

############## IK computing ####################################################################
def joint_lower_leg_ik(robot, oMdes_FL, JOINT_ID_FL, oMdes_FR, JOINT_ID_FR, oMdes_RL, JOINT_ID_RL, oMdes_RR, JOINT_ID_RR, Freebase, Homing_pose):
    ############ IK-solution ###############################################################33
    IK_FL_leg = CLIK(robot, oMdes_FL, JOINT_ID_FL, Freebase)
    IK_FR_leg = CLIK(robot, oMdes_FR, JOINT_ID_FR, Freebase)
    IK_RL_leg = CLIK(robot, oMdes_RL, JOINT_ID_RL, Freebase)
    IK_RR_leg = CLIK(robot, oMdes_RR, JOINT_ID_RR, Freebase)

    q = robot.q0
    if t < t_homing + 0.05:
        if Freebase:
            q[0 + 7:4 + 7] = Homing_pose[4:8]
            q[4 + 7:8 + 7] = Homing_pose[0:4]
            q[8 + 7:12 + 7] = Homing_pose[12:16]
            q[12 + 7:16 + 7] = Homing_pose[8:12]
        else:
            q = Homing_pose

    # ############### Jacobian-based IK: not used for quadrupedal#############################
    q_FL, J_FL = IK_FL_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_FR, J_FR = IK_FR_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_RL, J_RL = IK_RL_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_RR, J_RR = IK_RR_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    ############################################################
    ############## nonlinear optimization based IK solvers#############################
    q_FL = IK_FL_leg.fbgs_opt(q)
    q_FR = IK_FR_leg.fbgs_opt(q)
    q_RL = IK_RL_leg.fbgs_opt(q)
    q_RR = IK_RR_leg.fbgs_opt(q)


    ##### transfer the pinocchio joint to pybullet joint##########################
    q_ik = Homing_pose
    if Freebase:
        q_ik[0:4] = q_FR[4 + 7:8 + 7]
        q_ik[4:8] = q_FL[0 + 7:4 + 7]
        q_ik[8:12] = q_RR[12 + 7:16 + 7]
        q_ik[12:16] = q_RL[8 + 7:12 + 7]

    return q_ik


#################################################################################################
global base_homing
global pr_homing
global pl_homing

############################################################################### robot setup ###############
########whold-body simu:
Full_body_simu = True
##########for robot with float-base ################################
Freebase = True

mesh_dir = str(Path(__file__).parent.absolute())

# You should change here to set up your own URDF file
if Full_body_simu:
    # urdf_filename = mesh_dir + '/go1_description/urdf/go1_full_no_grippers.urdf'
    urdf_filename = mesh_dir + '/go1_description/urdf/go1_origin.urdf'
else:
    urdf_filename = mesh_dir + '/go1_description/urdf/go1_origin.urdf'

### pinocchio load urdf
if Freebase:
    robot =  RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir, pin.JointModelFreeFlyer())
    addFreeFlyerJointLimits(robot)
else:
    robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir)
## explore the model class
for name, function in robot.model.__class__.__dict__.items():
    print(' **** %s: %s' % (name, function.__doc__))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
print('standard model: dim=' + str(len(robot.model.joints)))
for jn in robot.model.joints:
    print(jn)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' )

# find lower-leg joint idx in pinocchio;
id_FR =['FR_hip_joint','FR_thigh_joint','FR_calf_joint','FR_foot_fixed']
id_FL =['FL_hip_joint','FL_thigh_joint','FL_calf_joint','FL_foot_fixed']
id_RR =['RR_hip_joint','RR_thigh_joint','RR_calf_joint','RR_foot_fixed']
id_RL =['RL_hip_joint','RL_thigh_joint','RL_calf_joint','RL_foot_fixed']
idFR =[]
idFL =[]
idRR =[]
idRL =[]

for i in range(0,len(id_FR)):
    idFR.append(robot.model.getJointId(id_FR[i]))
    idFL.append(robot.model.getJointId(id_FL[i]))
    idRR.append(robot.model.getJointId(id_RR[i]))
    idRL.append(robot.model.getJointId(id_RL[i]))
print("FR leg joint id in pinocchio",idFR)
print("FL leg joint id in pinocchio",idFL)
print("RR leg joint id in pinocchio",idRR)
print("RL leg joint id in pinocchio",idRL)

##### reset base pos and orn,  only workable in floating-based model
# robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array([1,0.5,0.8]))

q = robot.q0

print("q:", q)

############################### pinocchio load finish !!!!!!!!!!!!!!!!!!!!!!!!!!! #####################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio load urdf finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

################################################### pybullet simulation loading ###########################
### intial pose for go1 in pybullet
sim_rate = 200
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)

urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=[0, 0, 0.446],
                 baseRPY=[0, 0, 0],
                 Torquecontrol = False)
go1id = urobtx.id

num_joints = urobtx.getNumJoints()
num_actuated_joint = urobtx.getNumActuatedJoints()
actuation_joint_index = urobtx.getActuatedJointIndexes()

Homing_pose = np.zeros(num_actuated_joint)
### Homing_pose: four legs:FR, FL, RR, RL####### is important for walking in place
Homing_pose[0] =  0
Homing_pose[1] =  0.75
Homing_pose[2] =  -1.8
Homing_pose[3] = 0
Homing_pose[4] =  0.75
Homing_pose[5] =  -1.8
Homing_pose[6] = 0
Homing_pose[7] =  0.75
Homing_pose[8] =  -1.8
Homing_pose[9] =  0
Homing_pose[10] =  0.75
Homing_pose[11] =  -1.8


Homing_height_reduce = 0.1  ####bending knee

print("Homing_pose:",Homing_pose)


# q_initial = np.zeros(num_actuated_joint)
# for i in range(0, num_actuated_joint):
#     q_initial[i] = Homing_pose[i]*0.01
# urobtx.resetJointStates(q_initial)

q_cmd = np.zeros(num_actuated_joint)
q_cmd_pre = np.zeros(num_actuated_joint)
q_vel_cmd = np.zeros(num_actuated_joint)
torque_cmd = np.zeros(num_actuated_joint)
t_homing = 5
n_t_homing = round(t_homing/dt)

useRealTimeSimulation = 0
pybullet.setRealTimeSimulation(useRealTimeSimulation)
############################## enable FSR sensoring
idFR_fsr = idFR[2]+1
idFL_fsr = idFL[2]+1
idRR_fsr = idRR[2]+1
idRL_fsr = idRL[2]+1
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idFR_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idFL_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idRR_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idRL_fsr,enableSensor=1)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pybullet load environment finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")
######################################################3
##### Gait_Controller
Controller_ver = Gait_Controller(urbodx = urobtx, id = go1id,verbose=True)

trailDuration = 10
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0.446]
hasPrevPose = 0

t=0.
leg_FL_homing_fix = []
leg_FR_homing_fix = []
leg_RL_homing_fix = []
leg_RR_homing_fix = []
base_home_fix = []

torso1_linkid = 0
left_sole_linkid = 36
right_sole_linkid = 42



mesh_dirx = str(Path(__file__).parent.absolute())
# angle_cmd_filename = mesh_dirx + '/go1/go1_angle_nmpc.txt'
# torque_cmd_filename = mesh_dirx + '/go1/go1_ref_torque_nmpc.txt'
#
# joint_angle_cmd = np.loadtxt(angle_cmd_filename)  ### linux/ubuntu
# joint_torque_cmd = np.loadtxt(torque_cmd_filename)  ### linux/ubuntu
#
# print(type(joint_angle_cmd))
# row_num = joint_angle_cmd.shape[0]
# col_num = joint_angle_cmd.shape[1]
# print(col_num)


##############For kinematics;
des_FR_p = np.zeros([3,1])
des_FL_p = np.zeros([3,1])
des_RR_p = np.zeros([3,1])
des_RL_p = np.zeros([3,1])
des_FR_v = np.zeros([3,1])
des_FL_v = np.zeros([3,1])
des_RR_v = np.zeros([3,1])
des_RL_v = np.zeros([3,1])

### initialization for sim_robots, only once for FK
des_FL = np.array([3,1,1])
oMdes_FL = pin.SE3(np.eye(3), des_FL)
JOINT_ID_FL = idFL[-1]
#########leg IK ###############3
IK_leg = CLIK(robot, oMdes_FL, JOINT_ID_FL, Freebase)

###########gait planner
Gait_func = Gait(T=0.8, Sx = 0.05, Sy = 0, Sz = 0, lift = 0.02, T_num = 20, Dsp_ratio=0.1, dt = dt, Qudrupedal=True)

N_gait = Gait_func.step_location()
N_gait += n_t_homing



full_joint_number = urobtx.getNumJoints()
FileLength = N_gait
traj_opt = np.zeros([FileLength,12])  ### trajectory reference generated by gait planner
joint_opt = np.zeros([FileLength,num_actuated_joint])  #### joint angle by IK
state_feedback = np.zeros([FileLength,40])   ### robot state estimation
links_pos_prev = np.zeros([full_joint_number+1,3])  ### links com position; the last one is the base position
links_vel_prev = np.zeros([full_joint_number+1,3])  ### links com velocities
support_flag = np.zeros([FileLength,1])
gcom_pre = [0,0,0]
com_ref_base  = [0,0,0]
com_feedback_base  = [0,0,0]
com_ref_det = [0,0,0]
com_ref_det_pre = [0,0,0]
com_feedback_det= [0,0,0]
com_feedback_det_pre= [0,0,0]

angle_ref_det= [0,0,0]
angle_feedback_det= [0,0,0]
angle_ref_det_pre= [0,0,0]
angle_feedback_det_pre= [0,0,0]

# desired foot force
Force_FR = np.zeros([3,1])
Force_FL = np.zeros([3,1])
Force_RR = np.zeros([3,1])
Force_RL = np.zeros([3,1])
Force_FR_old = np.zeros([3,1])
Force_FL_old = np.zeros([3,1])
Force_RR_old = np.zeros([3,1])
Force_RL_old = np.zeros([3,1])

q_FR = np.zeros(3)
q_FL = np.zeros(3)
q_RR = np.zeros(3)
q_RL = np.zeros(3)
Q_cmd = np.zeros([num_actuated_joint,N_gait])
Q_measure = np.zeros([num_actuated_joint,N_gait])
Q_velocity_measure = np.zeros([num_actuated_joint,N_gait])

leg_FR_homing = np.zeros([3,1])
leg_FL_homing = np.zeros([3,1])
leg_RR_homing = np.zeros([3,1])
leg_RL_homing = np.zeros([3,1])
J_FR = np.zeros([3,3])
J_FL = np.zeros([3,3])
J_RR = np.zeros([3,3])
J_RL = np.zeros([3,3])
des_base = np.array([0, 0, 0.446])
des_base_vel = np.zeros(3)
h_com = 0
Torque_cmd = np.zeros([num_actuated_joint,N_gait])

det_torque_pd = np.zeros([num_actuated_joint,1])
############## Sample Gains
# joint PD gains
kp = np.array([100, 50, 50])
kd = np.array([1, 0.1, 0.1])*0.1
# Cartesian PD gains
kpCartesian = np.diag([2500] * 3)
kdCartesian = np.diag([40] * 3)

Torque_measured = np.zeros([num_actuated_joint,N_gait])
FSR_measured = np.zeros([num_actuated_joint,N_gait])
Torque_gravity = 0.8

#### relative to base framework
Relative_FR_pos = np.zeros(3)
Relative_FR_vel = np.zeros(3)
Relative_FL_pos = np.zeros(3)
Relative_FL_vel = np.zeros(3)
Relative_RR_pos = np.zeros(3)
Relative_RR_vel = np.zeros(3)
Relative_RL_pos = np.zeros(3)
Relative_RL_vel = np.zeros(3)

### for fsr tracking controller
Force_tracking_kd = 0.001 * np.ones(3)
Force_tracking_kd[2] = 0.001

#### impedance control
IP_tracking_kp = 300 * np.ones(3)
IP_tracking_kp[2] = 300
IP_tracking_kd = 20 * np.ones(3)
IP_tracking_kd[2] = 20
IP_Force_FR = np.zeros([3,1])
IP_Force_FL = np.zeros([3,1])
IP_Force_RR = np.zeros([3,1])
IP_Force_RL = np.zeros([3,1])
actuatedJointtorques = np.zeros(num_actuated_joint)
####################################### main loop for robot gait generation and control ####################
i = 1

tx = 0

while i<=3*n_t_homing:
    if (useRealTimeSimulation):
        t = t + dt
    else:
        t = t + dt


    if(i>1):
        Controller_ver.ankle_joint_pressure()
        q_mea = urobtx.getActuatedJointPositions()
        dq_mea = urobtx.getActuatedJointVelocities()
        T_mea = urobtx.getActuatedJointtorques()

        FR_fsr = np.array(Controller_ver.FR_sensor[0:3])
        FL_fsr = np.array(Controller_ver.FL_sensor[0:3])
        RR_fsr = np.array(Controller_ver.RR_sensor[0:3])
        RL_fsr = np.array(Controller_ver.RL_sensor[0:3])
        FSR_measured[0:3,i-1] = FR_fsr
        FSR_measured[3:6, i - 1] = FL_fsr
        FSR_measured[6:9,i-1] = RR_fsr
        FSR_measured[9:12, i - 1] = RL_fsr

        Q_measure[:, i - 1] = q_mea
        Q_velocity_measure[:, i - 1] = q_mea
        Torque_measured[:, i - 1] = T_mea

    ## stage 1:  homing pose initialization ###########
    if i<=n_t_homing:
        #### initial state for homing_pose
        Homing_pose_t = Homing_pose*math.sin(t/t_homing/2.*math.pi)
        ####### forward state calculation #############
        ##### 0,1,2,3: FR, FL, RR, RL
        base_pos_m = prevPose1
        body_R = np.zeros([3,1])
        if (i<=1):
            q_FR = Homing_pose_t[0:3]
            q_FL = Homing_pose_t[3:6]
            q_RR = Homing_pose_t[6:9]
            q_RL = Homing_pose_t[9:12]
            leg_FR_homing, J_FR = IK_leg.fk_close_form(base_pos_m, body_R, q_FR, 0)
            leg_FL_homing, J_FL = IK_leg.fk_close_form(base_pos_m, body_R, q_FL, 1)
            leg_RR_homing, J_RR = IK_leg.fk_close_form(base_pos_m, body_R, q_RR, 2)
            leg_RL_homing, J_RL = IK_leg.fk_close_form(base_pos_m, body_R, q_RL, 3)
            h_com = base_pos_m[2]-(leg_FR_homing[2,0] + leg_FL_homing[2,0] + leg_RR_homing[2,0] + leg_RL_homing[2,0])/4

        des_FR_p = leg_FR_homing
        des_FL_p = leg_FL_homing
        des_RR_p = leg_RR_homing
        des_RL_p = leg_RL_homing

        ##### CoM trajectory: height variation
        Homing_height_t = h_com - Homing_height_reduce * math.sin(t / t_homing / 2. * math.pi)
        Homing_height_velt = -(1/t_homing/2 * math.pi) * Homing_height_reduce * math.cos(t / t_homing / 2. * math.pi)
        Homing_height_acct = (1/t_homing/2*math.pi)**2 * Homing_height_reduce * math.sin(t / t_homing / 2. * math.pi)

        des_base[2] = (Homing_height_t +(leg_FR_homing[2,0]+leg_FL_homing[2,0]+leg_RR_homing[2,0]+leg_RL_homing[2,0])/4)
        des_base_vel[2] = Homing_height_velt
        ############ Ik################
        body_R_des = np.zeros([3, 1])
        q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max=15, lamda=0.55)
        q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
        q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max=15, lamda=0.55)
        q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
        Q_cmd[0:3,i-1] = q_FR[0:3]
        Q_cmd[3:6,i-1] = q_FL[0:3]
        Q_cmd[6:9,i-1] = q_RR[0:3]
        Q_cmd[9:12,i-1] = q_RL[0:3]

        ##################################### feedback control#######################################
        # ####################### torque command as stance foot ################################
        # ##### test: only vertical movement
        # Fx_total = 0
        # Fy_total = 0
        # Fz_total = Controller_ver.mass * (Controller_ver.g + Homing_height_acct)
        # Force_FR[2, 0] = Fz_total * ((des_base[0] - leg_RR_homing[0,0])/(leg_FR_homing[0,0] - leg_RR_homing[0,0])) /2
        # Force_RR[2, 0] = Fz_total * ((leg_FR_homing[0,0] - des_base[0])/(leg_FR_homing[0,0] - leg_RR_homing[0,0])) /2
        # Force_FL[2, 0] = Force_FR[2,0]
        # Force_RL[2, 0] = Force_RR[2,0]
        # # if(i>1):
        # #     Controller_ver.ankle_joint_pressure()
        # #     FR_fsr = np.array(Controller_ver.FR_sensor[0:3])
        # #     FL_fsr = np.array(Controller_ver.FL_sensor[0:3])
        # #     RR_fsr = np.array(Controller_ver.RR_sensor[0:3])
        # #     RL_fsr = np.array(Controller_ver.RL_sensor[0:3])
        # #     if(i % 20==0):
        # #         print("des_base:", des_base)
        # #         #### FSR tracking control###################
        # #         for j in range(0,3):
        # #             Force_FR[j, 0] += Force_tracking_kd[j] * (Force_FR_old[j, 0] - FR_fsr[j])
        # #             Force_FL[j, 0] += Force_tracking_kd[j] * (Force_FL_old[j, 0] - FL_fsr[j])
        # #             Force_RR[j, 0] += Force_tracking_kd[j] * (Force_RR_old[j, 0] - RR_fsr[j])
        # #             Force_RL[j, 0] += Force_tracking_kd[j] * (Force_RL_old[j, 0] - RL_fsr[j])
        #
        #
        # torque_FR = -np.dot(J_FR.T, Force_FR)
        # torque_FL = -np.dot(J_FL.T, Force_FL)
        # torque_RR = -np.dot(J_RR.T, Force_RR)
        # torque_RL = -np.dot(J_RL.T, Force_RL)

        # ####################### torque command as stance foot ################################
        # for j in range(0,3):
        #     Relative_FR_pos[j] = des_FR_p[j,0] - des_base[j]
        #     Relative_FR_vel[j] = des_FR_v[j,0] - des_base_vel[j]
        #     Relative_FL_pos[j] = des_FL_p[j,0] - des_base[j]
        #     Relative_FL_vel[j] = des_FL_v[j,0] - des_base_vel[j]
        #     Relative_RR_pos[j] = des_RR_p[j,0] - des_base[j]
        #     Relative_RR_vel[j] = des_RR_v[j,0] - des_base_vel[j]
        #     Relative_RL_pos[j] = des_RL_p[j,0] - des_base[j]
        #     Relative_RL_vel[j] = des_RL_v[j,0] - des_base_vel[j]
        #
        # if(i>1):
        #     for j in range(0, 3):
        #         IP_Force_FR[j, 0] = IP_tracking_kp[j] *(
        #                 Relative_FR_pos[j] -(links_pos[Controller_ver.FR_soleid, j] - links_pos[-1, j])) +IP_tracking_kd[j] *(
        #                 Relative_FR_vel[j] -(links_vel[Controller_ver.FR_soleid, j] - links_vel[-1, j]))
        #
        #         IP_Force_FL[j, 0] = IP_tracking_kp[j] * (
        #                 Relative_FL_pos[j] - (links_pos[Controller_ver.FL_soleid, j] - links_pos[-1, j])) +IP_tracking_kd[j]*(
        #                 Relative_FL_vel[j] - (links_vel[Controller_ver.FL_soleid, j] - links_vel[-1, j]))
        #
        #         IP_Force_RR[j, 0] = IP_tracking_kp[j] *(
        #                 Relative_RR_pos[j] -(links_pos[Controller_ver.RR_soleid, j] - links_pos[-1, j])) +IP_tracking_kd[j] *(
        #                 Relative_RR_vel[j] -(links_vel[Controller_ver.RR_soleid, j] - links_vel[-1, j]))
        #
        #         IP_Force_RL[j, 0] = IP_tracking_kp[j] * (
        #                 Relative_RL_pos[j] - (links_pos[Controller_ver.RL_soleid, j] - links_pos[-1, j])) +IP_tracking_kd[j]*(
        #                 Relative_RL_vel[j] - (links_vel[Controller_ver.RL_soleid, j] - links_vel[-1, j]))
        #
        # torque_FR = -np.dot(J_FR.T, IP_Force_FR)
        # torque_FL = -np.dot(J_FL.T, IP_Force_FL)
        # torque_RR = -np.dot(J_RR.T, IP_Force_RR)
        # torque_RL = -np.dot(J_RL.T, IP_Force_RL)

        # ######################## Forward compensation#########################################3
        # ### gravity compensation
        # torque_FR[0,0] += Torque_gravity/2
        # torque_RR[0,0] += Torque_gravity/2
        # torque_FL[0,0] += (-Torque_gravity)/2
        # torque_RL[0,0] += (-Torque_gravity)/2

        # ################################### send torque command #################################
        # Torque_cmd[0:3,i-1] = torque_FR[0:3,0]
        # Torque_cmd[3:6,i-1] = torque_FL[0:3, 0]
        # Torque_cmd[6:9,i-1] = torque_RR[0:3,0]
        # Torque_cmd[9:12,i-1] = torque_RL[0:3, 0]

        # control_mode
        # if((urobtx.controlMode =='positionControl')and(i<20)):
        if(urobtx.controlMode == 'positionControl'):
            if (i <= 10):  ## add external torques to maintain balance since starte from singular configuration
                pybullet.applyExternalTorque(go1id, 0, torqueObj=[0, -10, 0], flags=1)

            urobtx.setActuatedJointPositions(Q_cmd[:,i-1])
            q_cmd = Q_cmd[:,i-1]
        else:
            # if (i <= 10):  ## only for toruqe control
                # pybullet.applyExternalTorque(go1id, 0, torqueObj=[0.5, -2, 0], flags=1)
                ## pybullet.applyExternalForce(go1id, 0, forceObj=[0, 0, -100], posObj=[0, 0, 0], flags=1)
            # urobtx.setActuatedJointTorques(Torque_cmd[:, i - 1])

            Torque_cmd[:, i - 1] = torque_cmd
            urobtx.setActuatedJointTorques(torque_cmd)

        gcom, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_pos, base_angle = Controller_ver.cal_com_state()
        links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i,dt,links_pos_prev,links_vel_prev)
        links_pos_prev = links_pos
        links_vel_prev = links_vel

        support_flag[i] = 0  ### 0, double support, right support
        ###################state feedback ###########################################
        # gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        # Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)

        # state_feedback[i,0:3] = gcom_m
        # state_feedback[i, 3:6] = right_sole_pos
        # state_feedback[i, 6:9] = left_sole_pos
        # state_feedback[i, 9:15] = right_ankle_force
        # state_feedback[i, 15:21] = left_ankle_force
        # state_feedback[i, 21:24] = gcop_m
        # state_feedback[i, 24:27] = dcm_pos_m
        # state_feedback[i, 27:30] = com_vel_m
        # state_feedback[i, 30:33] = base_pos_m
        # state_feedback[i, 33:36] = base_angle_m

        # links_pos_prev = links_pos
        # links_vel_prev = links_vel
        # gcom_pre = gcom_m
        # com_feedback_base = gcom_m
        # com_ref_base = base_pos_m
    else:
        ij = i - n_t_homing
        torque_cmd = urobtx.getActuatedJointtorques()
        if(ij<50):
            urobtx.setActuatedJointPositions(q_cmd)
            Q_cmd[:, i - 1] = q_cmd
            Torque_cmd[:, i - 1] = torque_cmd
        else:
            if(ij<200):
                torque_cmd = np.zeros(num_actuated_joint)
                urobtx.setActuatedJointTorques(torque_cmd)

                des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]
                des_base = np.array(des_basex)

                links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i, dt, links_pos_prev, links_vel_prev)
                des_FR_p[:, 0] = (links_pos[Controller_ver.FR_soleid, :]).T
                des_FL_p[:, 0] = (links_pos[Controller_ver.FL_soleid, :]).T
                des_RR_p[:, 0] = (links_pos[Controller_ver.RR_soleid, :]).T
                des_RL_p[:, 0] = (links_pos[Controller_ver.RL_soleid, :]).T

                Q_cmd[:, i - 1] = q_cmd
                Torque_cmd[:, i - 1] = torque_cmd

            else:
                tx = t - (t_homing + 200 * dt)
                if(ij==200):
                    des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]
                    des_base = np.array(des_basex)
                    gcom, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_pos, base_angle = Controller_ver.cal_com_state()
                    links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i, dt, links_pos_prev, links_vel_prev)

                    des_FR_p[:, 0] = (links_pos[Controller_ver.FR_soleid, :]).T
                    des_FL_p[:, 0] = (links_pos[Controller_ver.FL_soleid, :]).T
                    des_RR_p[:, 0] = (links_pos[Controller_ver.RR_soleid, :]).T
                    des_RL_p[:, 0] = (links_pos[Controller_ver.RL_soleid, :]).T

                if(tx>t_homing):
                    Homing_height_t = 0.3
                    Homing_height_velt = 0
                    Homing_height_acct = 0
                else:
                    Homing_height_t = 0.3 * math.sin(tx / t_homing / 2. * math.pi)
                    Homing_height_velt = (1/t_homing/2*math.pi) * 0.3 * math.cos(
                        tx/t_homing/2. * math.pi)
                    Homing_height_acct = -(1/t_homing/2*math.pi)**2 * 0.3 * math.sin(
                        tx/t_homing/2. * math.pi)

                des_base[2] = des_basex[2] + Homing_height_t
                des_base_vel[2] = Homing_height_velt

                ########### Ik################
                body_R_des = np.zeros([3, 1])
                q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max =15, lamda=0.55)
                q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
                q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max =15, lamda=0.55)
                q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
                Q_cmd[0:3,i-1] = q_FR[0:3]
                Q_cmd[3:6,i-1] = q_FL[0:3]
                Q_cmd[6:9,i-1] = q_RR[0:3]
                Q_cmd[9:12,i-1] = q_RL[0:3]
                q_cmd = Q_cmd[:,i-1]

                #################################### feedback control#######################################
                ####################### torque command as stance foot ################################
                ##### test: only vertical movement
                Fx_total = 0
                Fy_total = 0
                Fz_total = Controller_ver.mass * (Controller_ver.g + Homing_height_acct)
                Force_FR[2, 0] = Fz_total * (
                            (des_base[0]  - des_RR_p[0, 0]) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                Force_RR[2, 0] = Fz_total * (
                            (des_FR_p[0, 0] - des_base[0]) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                Force_FL[2, 0] = Force_FR[2, 0]
                Force_RL[2, 0] = Force_RR[2, 0]

                # Force_FR[0, 0] = Fx_total * (
                #             (des_base[0] + Homing_height_t*0.2 - des_RR_p[0, 0]) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                # Force_RR[0, 0] = Fx_total * (
                #             (des_FR_p[0, 0] - (des_base[0]+Homing_height_t*0.2)) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                # Force_FL[0, 0] = Force_FR[0, 0]
                # Force_RL[0, 0] = Force_RR[0, 0]


                torque_FR = -np.dot(J_FR.T, Force_FR)
                torque_FL = -np.dot(J_FL.T, Force_FL)
                torque_RR = -np.dot(J_RR.T, Force_RR)
                torque_RL = -np.dot(J_RL.T, Force_RL)

                # # ######################## Forward compensation#########################################3
                # # ### gravity compensation
                torque_FR[0,0] += Torque_gravity
                torque_RR[0,0] += Torque_gravity
                torque_FL[0,0] += (-Torque_gravity)
                torque_RL[0,0] += (-Torque_gravity)

                ####
                q_vel_cmd = (q_cmd -q_cmd_pre)/dt
                for ijx in range(0,4):
                    det_torque_pd[3*ijx+0:3*ijx+3,0] = (kp * (Q_cmd[3*ijx+0:3*ijx+3,i-1] -
                                                               Q_measure[3*ijx+0:3*ijx+3,i-1])
                                                       - kd * (q_vel_cmd[3*ijx+0:3*ijx+3] -
                                                               Q_velocity_measure[3*ijx+0:3*ijx+3,i-1])).T


                # ################################### send torque command #################################
                Torque_cmd[0:3, i - 1] = torque_FR[0:3, 0] + det_torque_pd[0:3,0]
                Torque_cmd[3:6, i - 1] = torque_FL[0:3, 0] + det_torque_pd[3:6,0]
                Torque_cmd[6:9, i - 1] = torque_RR[0:3, 0] + det_torque_pd[6:9,0]
                Torque_cmd[9:12, i - 1] = torque_RL[0:3, 0] + det_torque_pd[9:12,0]

                torque_cmd = Torque_cmd[:, i - 1]

                # urobtx.setActuatedJointPositions(Q_cmd[:,i-1])
                urobtx.setActuatedJointTorques(torque_cmd)

    #     # ######## LIP-based  reference trajectory generation #############################
    #     if Freebase: #routine1: change the base position and orientation for pinocchio IK: time-cost process due to the redundant freedom
    #         ################# test: data format is used for pinocchio
    #         des_base = np.array([0,
    #                              0 * (math.sin((t - t_homing) * 50 * math.pi / 180)),
    #                              -0.03 * abs(math.sin((t - t_homing) * 50 * math.pi / 180))]) + np.array(base_home_fix)
    #         # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
    #         # des_FL = np.array(leg_FL_homing_fix)
    #         # oMdes_FL = pin.SE3(np.eye(3), des_FL)
    #         # des_FR = np.array(leg_FR_homing_fix)
    #         # oMdes_FR = pin.SE3(np.eye(3), des_FR)
    #         # des_RL = np.array(leg_RL_homing_fix)
    #         # oMdes_RL = pin.SE3(np.eye(3), des_RL)
    #         # des_RR = np.array(leg_RR_homing_fix)
    #         # oMdes_RR = pin.SE3(np.eye(3), des_RR)
    #
    #     # ## gait planning-based on LIP
    #     j_index = int(i - n_t_homing)
    #     # pos_base = Gait_func.Ref_com_lip_update(j_index,h_com)
    #     #
    #     # Rlfoot_pos = Gait_func.FootpR(j_index)
    #     #
    #     #
    #     # des_base = np.array(base_home_fix)
    #     # des_base[0] = base_home_fix[0] + pos_base[0,0]
    #     # des_base[1] = base_home_fix[1] + pos_base[1,0]*0.00
    #     # des_base[2] = base_home_fix[2] + pos_base[2,0]
    #     # #### pair:FR--RL-; pair:FL--RR
    #     # des_FR_p[0:3,0] = leg_FR_homing + Rlfoot_pos[0:3, 0]
    #     # des_RL_p[0:3,0] = leg_RL_homing + Rlfoot_pos[0:3, 0]
    #     # des_FL_p[0:3, 0] = leg_FL_homing + Rlfoot_pos[3:6, 0]
    #     # des_RR_p[0:3, 0] = leg_RR_homing + Rlfoot_pos[3:6, 0]
    #
    #     body_R_des = np.zeros([3, 1])
    #
    #     q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max =15, lamda=0.55)
    #     q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
    #     q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max =15, lamda=0.55)
    #     q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
    #
    #     ###
    #
    #     q_ik[0:3] = q_FR
    #     q_ik[4:7] = q_FL
    #     q_ik[8:11] = q_RR
    #     q_ik[12:15] = q_RL
    #
    #     urobtx.setActuatedJointPositions(q_ik)
    #
    #     # urobtx.setActuatedJointTorques(q_ik)
    #
    #
    #
        # if(i==5*n_t_homing):
        #     plt.figure()
    #         plt.figure()
    #         plt.subplot(3, 2, 1)
    #         plt.plot(Gait_func.comx)
    #         plt.plot(Gait_func.px)
    #         plt.subplot(3, 2, 2)
    #         plt.plot(Gait_func.comy)
    #         plt.plot(Gait_func.py)
    #         plt.subplot(3, 2, 3)
    #         plt.plot(Gait_func.comvx)
    #         plt.subplot(3, 2, 4)
    #         plt.plot(Gait_func.comvy)
    #         plt.subplot(3, 2, 5)
    #         plt.plot(Gait_func.comax)
    #         plt.subplot(3, 2, 6)
    #         plt.plot(Gait_func.comay)
    #         plt.show()
    #
    #         plt.figure()
    #         plt.subplot(2, 3, 1)
    #         plt.plot(Gait_func._Rfootx)
    #         plt.plot(Gait_func._Lfootx)
    #         plt.subplot(2, 3, 2)
    #         plt.plot(Gait_func._Rfooty)
    #         plt.plot(Gait_func._Lfooty)
    #         plt.subplot(2, 3, 3)
    #         plt.plot(Gait_func._Rfootz)
    #         plt.plot(Gait_func._Lfootz)
    #         plt.subplot(2, 3, 4)
    #         plt.plot(Gait_func._Rfootvx)
    #         plt.plot(Gait_func._Lfootvx)
    #         plt.subplot(2, 3, 5)
    #         plt.plot(Gait_func._Rfootvy)
    #         plt.plot(Gait_func._Lfootvy)
    #         plt.subplot(2, 3, 6)
    #         plt.plot(Gait_func._Rfootvz)
    #         plt.plot(Gait_func._Lfootvz)
    #         plt.show()

        # ############ IK-solution for the float-based humanod: providing initial guess "homing_pose" #######################
        # ########### set endeffector id for ik using pinocchio
        # JOINT_ID_FL = idFL[-1]
        # JOINT_ID_FR = idFR[-1]
        # JOINT_ID_RL = idRL[-1]
        # JOINT_ID_RR = idRR[-1]
        #
        # q_ik = joint_lower_leg_ik(robot, oMdes_FL, JOINT_ID_FL, oMdes_FR, JOINT_ID_FR, oMdes_RL, JOINT_ID_RL, oMdes_RR, JOINT_ID_RR, Freebase, Homing_pose)
        #
        # if (i % 10 == 0):
        #     print("q_ik_cmd:", q_ik)
        # ######## joint command: position control mode ###########################
        # joint_opt[i] = q_ik
        # urobtx.setActuatedJointPositions(q_ik)



        # ###################state feedback ###########################################
        # gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        # Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)
        #
        # state_feedback[i,0:3] = gcom_m
        # state_feedback[i, 3:6] = right_sole_pos
        # state_feedback[i, 6:9] = left_sole_pos
        # state_feedback[i, 9:15] = right_ankle_force
        # state_feedback[i, 15:21] = left_ankle_force
        # state_feedback[i, 21:24] = gcop_m
        # state_feedback[i, 24:27] = dcm_pos_m
        # state_feedback[i, 27:30] = com_vel_m
        # state_feedback[i, 30:33] = base_pos_m
        # state_feedback[i, 33:36] = base_angle_m
        #
        # links_pos_prev = links_pos
        # links_vel_prev = links_vel
        # gcom_pre = gcom_m
        #
        # if ((abs(base_angle_m[0]) >=20* math.pi / 180) or (abs(base_angle_m[1]) >=20* math.pi / 180) ): ### falling down
        #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_torque_nmpc.txt', Torque_cmd,fmt='%s', newline='\n')
        #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_angle_nmpc.txt', Torque_cmd,fmt='%s', newline='\n')
        #     # np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_state_est_nmpc.txt', state_feedback,fmt='%s', newline='\n')

        # ################## IK-based control: in this case, we can use admittance control, preview control and PD controller for CoM control #################################33
        # com_ref_det = np.array(des_base) - np.array(com_ref_base)
        # com_feedback_det = np.array(gcom_m) - np.array(com_feedback_base)
        # angle_ref_det = des_base_ori
        # angle_feedback_det = base_angle_m
        #
        # det_comxxxx, det_body_anglexxxx =Controller_ver.CoM_Body_pd(dt,com_ref_det, com_feedback_det, com_ref_det_pre, com_feedback_det_pre, angle_ref_det,angle_feedback_det, angle_ref_det_pre, angle_feedback_det_pre)
        # des_com_pos_control = det_comxxxx + np.array(des_base)
        # det_base_angle_control = det_body_anglexxxx
        # det_base_matrix_control = Controller_ver.RotMatrixfromEuler(det_base_angle_control)
        # robot.model.jointPlacements[1] = pin.SE3(det_base_matrix_control, des_com_pos_control)
        # # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_com_pos_control)
        #
        # com_ref_det_pre = com_ref_det
        # com_feedback_det_pre = com_feedback_det
        # angle_ref_det_pre = angle_ref_det
        # angle_feedback_det_pre = angle_feedback_det
        #

        # ###########################===========================================================
        # ######################################################################################

    Force_FR_old = Force_FR
    Force_FL_old = Force_FL
    Force_RR_old = Force_RR
    Force_RL_old = Force_RL
    links_pos_prev = links_pos
    links_vel_prev = links_vel
    q_cmd_pre = q_cmd

    if (i==3*n_t_homing):  ### save data
        np.savetxt('/home/jiatao/locomotion_nmpc_pybullet_python/go1/go1_measure_fsr.txt', FSR_measured,
                   fmt='%s',newline='\n')
        np.savetxt('/home/jiatao/locomotion_nmpc_pybullet_python/go1/go1_angle_ref.txt', Q_cmd, fmt='%s', newline='\n')
        np.savetxt('/home/jiatao/locomotion_nmpc_pybullet_python/go1/go1_angle_mea.txt', Q_measure, fmt='%s', newline='\n')
        np.savetxt('/home/jiatao/locomotion_nmpc_pybullet_python/go1/go1_torque_ref.txt', Torque_cmd, fmt='%s',
                   newline='\n')
        np.savetxt('/home/jiatao/locomotion_nmpc_pybullet_python/go1/go1_torque_mea.txt', Torque_measured, fmt='%s',
                   newline='\n')

    i += 1

    ##### doesn't use it in realtime simu mode
    #pybullet.stepSimulation()
    # time.sleep(dt)
    sim_env.step()

    ##############only work in real-time model???? a bug #########################3
    # ls = pybullet.getLinkState(go1id, torso1_linkid)
    # print("torso_link:",ls[0])
    # if (hasPrevPose):
    #     # pybullet.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
    #     pybullet.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    # # prevPose = lsx_ori
    # prevPose1 = ls[4]
    # hasPrevPose = 1
