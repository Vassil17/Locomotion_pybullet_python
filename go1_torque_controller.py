#### python envir setup
from __future__ import print_function

import copy
import os
from os.path import dirname, join, abspath
import sys
import platform

from pathlib import Path

### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                        #### IK solver
from robot_tracking_controller import Gait_Controller #### State estimate
from LIP_motion_planner import Gait                   #### Gait planner
from robot_dynamics import ForceCtrl                  #### Force controller

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
print("mesh_dir:",mesh_dir)

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

####### only for debug check #########################################
# ## explore the model class
# for name, function in robot.model.__class__.__dict__.items():
#     print(' **** %s: %s' % (name, function.__doc__))
# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
# print('standard model: dim=' + str(len(robot.model.joints)))
# for jn in robot.model.joints:
#     print(jn)
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
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio setup finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

################################################### pybullet simulation setup ###########################
### intial pose for go1 in pybullet
sim_rate = 200
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)

trailDuration = 10
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0.446]
hasPrevPose = 0
urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=prevPose1,
                 baseRPY=[0, 0, 0],
                 Torquecontrol = False)
go1id = urobtx.id

num_joints = urobtx.getNumJoints()
num_actuated_joint = urobtx.getNumActuatedJoints()
joint_perLeg = int(num_actuated_joint/4.0)
actuation_joint_index = urobtx.getActuatedJointIndexes()

### Homing_pose: four legs:FR, FL, RR, RL####### is important for walking in place
Homing_pose = np.zeros(num_actuated_joint)
for jy in range(0,4):
    Homing_pose[jy * joint_perLeg + 0] = 0
    Homing_pose[jy * joint_perLeg + 1] = 0.75
    Homing_pose[jy * joint_perLeg + 2] = -1.8
print("Homing_pose:",Homing_pose)

Homing_height_reduce = 0.1  ####bending knee

q_cmd = np.zeros(num_actuated_joint)
q_cmd_pre = np.zeros(num_actuated_joint)
q_vel_cmd = np.zeros(num_actuated_joint)
torque_cmd = np.zeros(num_actuated_joint)
t_homing = 5
n_t_homing = round(t_homing/dt)

useRealTimeSimulation = 0
pybullet.setRealTimeSimulation(useRealTimeSimulation)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pybullet load environment finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

######################################################
##### Gait_Controller_estimation
State_estimator = Gait_Controller(urbodx = urobtx, id = go1id,verbose=True)
##### Force controller
Force_controller = ForceCtrl(urbodx = urobtx)

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


##############For kinematics
### desired leg position and velocity ##############################
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
base_R = np.zeros([3,1])
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
Gravity_comp = np.zeros(num_actuated_joint)
Gravity_comp[0] = Torque_gravity
Gravity_comp[joint_perLeg] = Torque_gravity
Gravity_comp[2*joint_perLeg] = -Torque_gravity
Gravity_comp[3*joint_perLeg] = -Torque_gravity
#### relative to base framework
Relative_FR_pos = np.zeros(3)
Relative_FR_vel = np.zeros(3)
Relative_FL_pos = np.zeros(3)
Relative_FL_vel = np.zeros(3)
Relative_RR_pos = np.zeros(3)
Relative_RR_vel = np.zeros(3)
Relative_RL_pos = np.zeros(3)
Relative_RL_vel = np.zeros(3)
Relative_FR_pos_mea = np.zeros(3)
Relative_FR_vel_mea = np.zeros(3)
Relative_FL_pos_mea = np.zeros(3)
Relative_FL_vel_mea = np.zeros(3)
Relative_RR_pos_mea = np.zeros(3)
Relative_RR_vel_mea = np.zeros(3)
Relative_RL_pos_mea = np.zeros(3)
Relative_RL_vel_mea = np.zeros(3)
### foot support stance
FR_support = True
FL_support = True
RR_support = True
RL_support = True

#####

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
    #################################### state esimation ##############################################################
    gcom, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_pos, base_angle = State_estimator.cal_com_state()
    links_pos, links_vel, links_acc = State_estimator.get_link_vel_vol(i,dt,links_pos_prev,links_vel_prev)


    State_estimator.ankle_joint_pressure()
    q_mea = urobtx.getActuatedJointPositions()
    dq_mea = urobtx.getActuatedJointVelocities()
    T_mea = urobtx.getActuatedJointtorques()

    bas_pos,bas_ori = pybullet.getBasePositionAndOrientation(go1id)
    measure_base = np.array(bas_pos)
    bas_eular = pybullet.getEulerFromQuaternion(bas_ori)
    for jx in range(0,3):
        base_R[jx,0] = base_angle[jx]
    q_FR = q_mea[0:3]
    q_FL = q_mea[3:6]
    q_RR = q_mea[6:9]
    q_RL = q_mea[9:12]
    leg_FR, J_FR = IK_leg.fk_close_form(measure_base, base_R, q_FR, 0)
    leg_FL, J_FL = IK_leg.fk_close_form(measure_base, base_R, q_FL, 1)
    leg_RR, J_RR = IK_leg.fk_close_form(measure_base, base_R, q_RR, 2)
    leg_RL, J_RL = IK_leg.fk_close_form(measure_base, base_R, q_RL, 3)

    for j in range(0, 3):
        Relative_FR_pos_mea[j] = links_pos[State_estimator.FR_soleid, j] - links_pos[-1, j]
        Relative_FR_vel_mea[j] = links_vel[State_estimator.FR_soleid, j] - links_vel[-1, j]
        Relative_FL_pos_mea[j] = links_pos[State_estimator.FL_soleid, j] - links_pos[-1, j]
        Relative_FL_vel_mea[j] = links_vel[State_estimator.FL_soleid, j] - links_vel[-1, j]
        Relative_RR_pos_mea[j] = links_pos[State_estimator.RR_soleid, j] - links_pos[-1, j]
        Relative_RR_vel_mea[j] = links_vel[State_estimator.RR_soleid, j] - links_vel[-1, j]
        Relative_RL_pos_mea[j] = links_pos[State_estimator.RL_soleid, j] - links_pos[-1, j]
        Relative_RL_vel_mea[j] = links_vel[State_estimator.RL_soleid, j] - links_vel[-1, j]

    FR_fsr = np.array(State_estimator.FR_sensor[0:3])
    FL_fsr = np.array(State_estimator.FL_sensor[0:3])
    RR_fsr = np.array(State_estimator.RR_sensor[0:3])
    RL_fsr = np.array(State_estimator.RL_sensor[0:3])
    FSR_measured[0:3,i-1] = FR_fsr
    FSR_measured[3:6,i-1] = FL_fsr
    FSR_measured[6:9,i-1] = RR_fsr
    FSR_measured[9:12,i-1] = RL_fsr

    Q_measure[:, i - 1] = q_mea
    Q_velocity_measure[:, i - 1] = q_mea
    Torque_measured[:, i - 1] = T_mea

    ## stage 1:  homing pose initialization ###########
    if i<=n_t_homing:
        #### initial state for homing_pose
        Homing_pose_t = Homing_pose*math.sin(t/t_homing/2.*math.pi)
        ####### forward state calculation #############
        base_pos_m = prevPose1
        body_R = np.zeros([3,1])
        if (i<=1): ####calculate the inital position of four legs
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

        ##################################### feedback control preparation#######################################
        for j in range(0,3):
            Relative_FR_pos[j] = des_FR_p[j,0] - des_base[j]
            Relative_FR_vel[j] = des_FR_v[j,0] - des_base_vel[j]
            Relative_FL_pos[j] = des_FL_p[j,0] - des_base[j]
            Relative_FL_vel[j] = des_FL_v[j,0] - des_base_vel[j]
            Relative_RR_pos[j] = des_RR_p[j,0] - des_base[j]
            Relative_RR_vel[j] = des_RR_v[j,0] - des_base_vel[j]
            Relative_RL_pos[j] = des_RL_p[j,0] - des_base[j]
            Relative_RL_vel[j] = des_RL_v[j,0] - des_base_vel[j]

        # control_mode: should be  positionControl mode
        if(urobtx.controlMode == 'positionControl'):
            if (i <= 10):  ## add external torques to maintain balance since starte from singular configuration
                pybullet.applyExternalTorque(go1id, 0, torqueObj=[0, -10, 0], flags=1)

            urobtx.setActuatedJointPositions(Q_cmd[:,i-1])
            q_cmd = Q_cmd[:,i-1]
        else:
            Torque_cmd[:, i - 1] = torque_cmd
            urobtx.setActuatedJointTorques(torque_cmd)

        support_flag[i] = 0  ### 0, double support, right support

    else:
        ij = i - n_t_homing
        torque_cmd = urobtx.getActuatedJointtorques()
        if(ij<50):
            urobtx.setActuatedJointPositions(q_cmd)
            Q_cmd[:, i - 1] = q_cmd
            Torque_cmd[:, i - 1] = torque_cmd
        else:
            if(ij<200):#####falling down
                torque_cmd = np.zeros(num_actuated_joint)
                urobtx.setActuatedJointTorques(torque_cmd)

                des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]
                des_base = np.array(des_basex)

                des_FR_p[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                des_FL_p[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                des_RR_p[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                des_RL_p[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T

                Q_cmd[:, i - 1] = q_cmd
                Torque_cmd[:, i - 1] = torque_cmd
            else: ######switching to torque control mode
                tx = t - (t_homing + 200 * dt)

                ########gait planning #######################
                if(ij==200):
                    des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]
                    des_base = np.array(des_basex)
                    des_FR_p[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                    des_FL_p[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                    des_RR_p[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                    des_RL_p[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T

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


                for j in range(0, 3):
                    Relative_FR_pos[j] = des_FR_p[j, 0] - des_base[j]
                    Relative_FR_vel[j] = des_FR_v[j, 0] - des_base_vel[j]
                    Relative_FL_pos[j] = des_FL_p[j, 0] - des_base[j]
                    Relative_FL_vel[j] = des_FL_v[j, 0] - des_base_vel[j]
                    Relative_RR_pos[j] = des_RR_p[j, 0] - des_base[j]
                    Relative_RR_vel[j] = des_RR_v[j, 0] - des_base_vel[j]
                    Relative_RL_pos[j] = des_RL_p[j, 0] - des_base[j]
                    Relative_RL_vel[j] = des_RL_v[j, 0] - des_base_vel[j]

                ########### Ik ###############################
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

                q_vel_cmd = (q_cmd -q_cmd_pre)/dt
                ####################### For distribution ################################
                ##### test: only vertical movement
                Fx_total = 0
                Fy_total = 0
                Fz_total = State_estimator.mass * (State_estimator.g + Homing_height_acct)
                Force_FR[2, 0] = Fz_total * (
                            (des_base[0]  - des_RR_p[0, 0]) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                Force_RR[2, 0] = Fz_total * (
                            (des_FR_p[0, 0] - des_base[0]) / (des_FR_p[0, 0] - des_RR_p[0, 0])) / 2
                Force_FL[2, 0] = Force_FR[2, 0]
                Force_RL[2, 0] = Force_RR[2, 0]

                ###############
                torque_FR = Force_controller.torque_cmd(FR_support,J_FR, Force_FR,q_cmd[0:3], q_mea[0:3], q_vel_cmd[0:3], dq_mea[0:3],
                                                        Relative_FR_pos[0:3], Relative_FR_pos_mea[0:3], Relative_FR_vel[0:3], Relative_FR_vel_mea[0:3],Gravity_comp[0:3])
                torque_FL = Force_controller.torque_cmd(FL_support,J_FL, Force_FL,q_cmd[3:6], q_mea[3:6], q_vel_cmd[3:6], dq_mea[3:6],
                                                        Relative_FR_pos[3:6], Relative_FR_pos_mea[3:6], Relative_FR_vel[3:6], Relative_FR_vel_mea[3:6],Gravity_comp[3:6])
                torque_RR = Force_controller.torque_cmd(RR_support,J_RR, Force_RR,q_cmd[6:9], q_mea[6:9], q_vel_cmd[6:9], dq_mea[6:9],
                                                        Relative_FR_pos[6:9], Relative_FR_pos_mea[6:9], Relative_FR_vel[6:9], Relative_FR_vel_mea[6:9],Gravity_comp[6:9])
                torque_RL = Force_controller.torque_cmd(FL_support,J_RL, Force_RL,q_cmd[9:12], q_mea[9:12], q_vel_cmd[9:12], dq_mea[9:12],
                                                        Relative_FR_pos[9:12], Relative_FR_pos_mea[9:12], Relative_FR_vel[9:12], Relative_FR_vel_mea[9:12],Gravity_comp[9:12])
                # ################################### send torque command #################################
                Torque_cmd[0:3, i - 1] = torque_FR[0:3]
                Torque_cmd[3:6, i - 1] = torque_FL[0:3]
                Torque_cmd[6:9, i - 1] = torque_RR[0:3]
                Torque_cmd[9:12, i - 1] = torque_RL[0:3]

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


    Force_FR_old = Force_FR
    Force_FL_old = Force_FL
    Force_RR_old = Force_RR
    Force_RL_old = Force_RL
    links_pos_prev = links_pos
    links_vel_prev = links_vel
    q_cmd_pre = q_cmd


    if (i==3*n_t_homing):  ### save data

        fsr_mea_dir = mesh_dir + '/go1/go1_measure_fsr.txt'
        angle_ref_dir = mesh_dir + '/go1/go1_angle_ref.txt'
        angle_mea_dir = mesh_dir + '/go1/go1_angle_mea.txt'
        torque_ref_dir = mesh_dir + '/go1/go1_torque_ref.txt'
        torque_mea_dir = mesh_dir + '/go1/go1_torque_mea.txt'

        np.savetxt(fsr_mea_dir, FSR_measured,
                   fmt='%s', newline='\n')
        np.savetxt(angle_ref_dir, Q_cmd, fmt='%s',
                   newline='\n')
        np.savetxt(angle_mea_dir, Q_measure, fmt='%s',
                   newline='\n')
        np.savetxt(torque_ref_dir, Torque_cmd, fmt='%s',
                   newline='\n')
        np.savetxt(torque_mea_dir, Torque_measured, fmt='%s',
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
