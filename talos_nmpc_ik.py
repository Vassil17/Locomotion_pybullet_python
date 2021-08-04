#### python envir setup
from __future__ import print_function

import os
from os.path import dirname, join, abspath
import sys

from pathlib import Path


### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                  #### IK solver
from robot_tracking_controller import Gait_Controller #### controller

##### numpy
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve


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

############################################## main loop function definition ###########################################
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

############## IK computing
def joint_lower_leg_ik(robot,oMdesl,JOINT_IDl,oMdesr,JOINT_IDr,Freebase,Homing_pose):
    ############ IK-solution ###############################################################33
    IK_left_leg = CLIK(robot, oMdesl, JOINT_IDl, Freebase)
    IK_right_leg = CLIK(robot, oMdesr, JOINT_IDr, Freebase)

    q = robot.q0
    if t < t_homing + 0.05:
        if Freebase:
            q[0 + 7:6 + 7] = Homing_pose[-12:-6]
            q[6 + 7:12 + 7] = Homing_pose[-6:]
        else:
            q = Homing_pose

    ############### Jacobian-based IK
    qr, Jr = IK_right_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    ql, Jl = IK_left_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    ############################################################
    #############################################################

    ##### transfer the pinocchio joint to pybullet joint##########################
    q_ik = Homing_pose
    if Freebase:
        q_ik[-12:-6] = ql[0 + 7:6 + 7]
        q_ik[-6:] = qr[6 + 7:12 + 7]

    else:
        q_ik[-12:-6] = ql[0:6]
        q_ik[-6:] = qr[6:12]
    return q_ik


#################################################################################################
global base_homing
global pr_homing
global pl_homing

############################################################################### robot setup ###############
########whold-body simu:
Full_body_simu = True
##########for robot with float-base: humanoids or ################################
Freebase = True

#mesh_dir = '/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos'
mesh_dir = str(Path(__file__).parent.absolute())

# You should change here to set up your own URDF file
if Full_body_simu:
    # urdf_filename = mesh_dir + '/talos_description/urdf/talos_full_no_grippers.urdf'
    urdf_filename = mesh_dir + '/talos_description/urdf/talos_full_no_grippers_bigger_plate.urdf'
else:
    urdf_filename = mesh_dir + '/talos_description/urdf/talos_lower_body_mesh_updated.urdf'

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

# find lower-leg joint idx in pinocchio
joints_desired_l = ['leg_left_1_joint','leg_left_2_joint','leg_left_3_joint', 'leg_left_4_joint', 'leg_left_5_joint','leg_left_6_joint']
joints_desired_r = ['leg_right_1_joint','leg_right_2_joint','leg_right_3_joint', 'leg_right_4_joint', 'leg_right_5_joint','leg_right_6_joint']
idr =[]
idl =[]
for i in range(0,len(joints_desired_r)):
    idl.append(robot.model.getJointId(joints_desired_l[i]))
    idr.append(robot.model.getJointId(joints_desired_r[i]))
print("left leg joint id in pinocchio",idl)
print("right leg joint id in pinocchio",idr)

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
############################### pinocchio load finish !!!!!!!!!!!!!!!!!!!!!!!!!!! #####################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio load urdf finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

################################################### pybullet simulation loading ###########################
### intial pose for talos in pybullet
###  by default, stepsimulation() is used
sim_rate = 200
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)

urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=[0, 0, 1.08],
                 baseRPY=[0, 0, 0])
talosid = urobtx.id


num_joint = urobtx.getNumActuatedJoints()
if Full_body_simu:
    Homing_pose = np.zeros(num_joint)
    ### upper arm
    Homing_pose[4] =  0.2
    Homing_pose[5] =  0.1
    Homing_pose[6] = -0.7
    Homing_pose[7] =  -0.6
    Homing_pose[11] =  -0.2
    Homing_pose[12] =  -0.1
    Homing_pose[13] = 0.7
    Homing_pose[14] =  -0.6
    ## lower leg
    Homing_pose[-10] =  -0.17644
    Homing_pose[-9] =  0.36913
    Homing_pose[-8] =  -0.18599
    Homing_pose[-4] =  -0.17644
    Homing_pose[-3] =  0.36913
    Homing_pose[-2] =  -0.18599
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
t_homing = 2

useRealTimeSimulation = 0
pybullet.setRealTimeSimulation(useRealTimeSimulation)

############################## enable ankle pressure sensoring
full_joint_number = urobtx.getNumJoints()
rig_legid = full_joint_number-2
left_legid = full_joint_number-9
pybullet.enableJointForceTorqueSensor(bodyUniqueId=talosid,jointIndex=rig_legid,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=talosid,jointIndex=left_legid,enableSensor=1)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pybullet load environment finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")
######################################################3

##### Gait_Controller
Controller_ver = Gait_Controller(urbodx = urobtx, id = talosid,verbose=True)


trailDuration = 10
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 1.08]
hasPrevPose = 0



t=0.
i=0
pr_homing_fix = []
pl_homing_fix = []
base_home_fix = []

torso1_linkid = 0
left_sole_linkid = 36
right_sole_linkid = 42


FileLength = 5000
traj_opt = np.zeros([FileLength,12])  ### trajectory reference generated by gait planner
joint_opt = np.zeros([FileLength,num_joint])  #### joint angle by IK
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

############################################################################################# main loop for robot gait generation and control ##########################################
while i<FileLength:
    if (useRealTimeSimulation):
        # dt = datetime.now()
        # t = (dt.second / 60.) * 2. * math.pi
        t = t + dt
    else:
        t = t + dt

    #################===============================================================================
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

        joint_opt[i] = Homing_pose_t
        urobtx.setActuatedJointPositions(Homing_pose_t)
        base_pos_m = pybullet.getBasePositionAndOrientation(talosid)[0]
        # left_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=6)[4]
        # right_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=13)[4]
        # base_homing = np.array(lsx_ori)
        if Freebase:
            robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array(base_pos_m))
        pin.forwardKinematics(robot.model,robot.data,q)
        pr_homing = robot.data.oMi[idr[5]].translation
        pl_homing = robot.data.oMi[idl[5]].translation
        base_home_fix = base_pos_m
        pl_homing_fix = tuple(pl_homing)
        pr_homing_fix = tuple(pr_homing)

        links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i,dt,links_pos_prev,links_vel_prev)
        links_pos_prev = links_pos
        links_vel_prev = links_vel

        support_flag[i] = 0  ### 0, double support, right support

        ###################state feedbacke ###########################################
        gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)

        state_feedback[i,0:3] = gcom_m
        state_feedback[i, 3:6] = right_sole_pos
        state_feedback[i, 6:9] = left_sole_pos
        state_feedback[i, 9:15] = right_ankle_force
        state_feedback[i, 15:21] = left_ankle_force
        state_feedback[i, 21:24] = gcop_m
        state_feedback[i, 24:27] = dcm_pos_m
        state_feedback[i, 27:30] = com_vel_m
        state_feedback[i, 30:33] = base_pos_m
        state_feedback[i, 33:36] = base_angle_m

        links_pos_prev = links_pos
        links_vel_prev = links_vel
        gcom_pre = gcom_m
        com_feedback_base = gcom_m
        com_ref_base = base_pos_m
    else:
        ###################state feedbacke ###########################################
        gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)

        state_feedback[i,0:3] = gcom_m
        state_feedback[i, 3:6] = right_sole_pos
        state_feedback[i, 6:9] = left_sole_pos
        state_feedback[i, 9:15] = right_ankle_force
        state_feedback[i, 15:21] = left_ankle_force
        state_feedback[i, 21:24] = gcop_m
        state_feedback[i, 24:27] = dcm_pos_m
        state_feedback[i, 27:30] = com_vel_m
        state_feedback[i, 30:33] = base_pos_m
        state_feedback[i, 33:36] = base_angle_m

        links_pos_prev = links_pos
        links_vel_prev = links_vel
        gcom_pre = gcom_m

        if ((abs(base_angle_m[0]) >=20* math.pi / 180) or (abs(base_angle_m[1]) >=20* math.pi / 180) ): ### falling down
            np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_traj_nmpc.txt', traj_opt,fmt='%s', newline='\n')
            np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_angle_nmpc.txt', joint_opt,fmt='%s', newline='\n')
            np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_state_est_nmpc.txt', state_feedback,fmt='%s', newline='\n')

        ######## reference trajectory generation #############################
        if Freebase: #routine1: change the base position and orientation for pinocchio IK: time-cost process due to the redundant freedom
            ################## test
            # des_base = np.array([0,
            #                    0.05 * (math.sin((t - t_homing) * 50 * math.pi / 180)),
            #                    -0.05 * abs(math.sin((t - t_homing) * 50 * math.pi / 180))]) + np.array(base_home_fix)
            # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
            # des_pl = np.array(pl_homing_fix)
            # oMdesl = pin.SE3(np.eye(3), des_pl)
            # des_pr = np.array(pr_homing_fix)
            # oMdesr = pin.SE3(np.eye(3), des_pr)
            ################################################### NMPC gait generation #####################################################
            j = i + 1- round(t_homing/dt)
            cpptest = "/home/jiatao/Dropbox/nmpc_pybullet/build/src/MPC_WALK.exe"  # in linux without suffix .exe
            donser = run_nmpc_external_ext(j,cpptest)

            if len(donser) == 12:
                traj_opt[i] = donser
            else:
                donser = traj_opt[i-1]
            des_base = donser[0:3]
            des_base_ori = donser[3:6]/10
            xxx = Controller_ver.RotMatrixfromEuler(des_base_ori)
            # robot.model.jointPlacements[1] = pin.SE3(xxx, des_base)
            robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
            des_pr = donser[6:9]
            des_pl = donser[9:12]

            if (i<=1):
                com_ref_base = des_base
                com_feedback_base = gcom_m

        else:  ####routine2: set the based position in local framework(note that always zeros), transform the base function in the
            des_pl = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pl_homing_fix)
            des_pr = np.array([0.03 * abs(math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0 - 0.05 * (math.sin((t - t_homing) * 5 * math.pi / 180)),
                               0.05 * abs(math.sin((t - t_homing) * 5 * math.pi / 180))]) + np.array(pr_homing_fix)
        ################## IK-based control: in this case, we can use admittance control, preview control and PD controller for CoM control #################################33
        com_ref_det = np.array(des_base) - np.array(com_ref_base)
        com_feedback_det = np.array(gcom_m) - np.array(com_feedback_base)
        angle_ref_det = des_base_ori
        angle_feedback_det = base_angle_m

        det_comxxxx, det_body_anglexxxx =Controller_ver.CoM_Body_pd(dt,com_ref_det, com_feedback_det, com_ref_det_pre, com_feedback_det_pre, angle_ref_det,angle_feedback_det, angle_ref_det_pre, angle_feedback_det_pre)
        des_com_pos_control = det_comxxxx + np.array(des_base)
        det_base_angle_control = det_body_anglexxxx
        det_base_matrix_control = Controller_ver.RotMatrixfromEuler(det_base_angle_control)
        robot.model.jointPlacements[1] = pin.SE3(det_base_matrix_control, des_com_pos_control)
        # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_com_pos_control)

        com_ref_det_pre = com_ref_det
        com_feedback_det_pre = com_feedback_det
        angle_ref_det_pre = angle_ref_det
        angle_feedback_det_pre = angle_feedback_det

        ############ IK-solution for the float-based humanod: providing initial guess "homing_pose" ###############################################################
        ########### set endeffector id for ik using pinocchio
        JOINT_IDl = idl[5]
        JOINT_IDr = idr[5]
        oMdesl = pin.SE3(np.eye(3), des_pl)
        oMdesr = pin.SE3(np.eye(3), des_pr)
        q_ik = joint_lower_leg_ik(robot, oMdesl, JOINT_IDl, oMdesr, JOINT_IDr, Freebase, Homing_pose)

        ######## joint command: position control mode ###########################
        joint_opt[i] = q_ik
        urobtx.setActuatedJointPositions(q_ik)

        ###########################===========================================================
        #########################################################################################

    i += 1
    if (i==FileLength-1):
        np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_traj_nmpc.txt',traj_opt,fmt='%s',newline='\n')
        np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_angle_nmpc.txt', joint_opt,fmt='%s', newline='\n')
        np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/talos/talos_state_est_nmpc.txt', state_feedback,fmt='%s', newline='\n')
    ##### doesn't use it in realtime simu mode
    pybullet.stepSimulation()

    ##############only work in real-time model???? a bug #########################3
    # ls = pybullet.getLinkState(talosid, torso1_linkid)
    # print("torso_link:",ls[0])
    # if (hasPrevPose):
    #     # pybullet.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
    #     pybullet.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    # # prevPose = lsx_ori
    # prevPose1 = ls[4]
    # hasPrevPose = 1
