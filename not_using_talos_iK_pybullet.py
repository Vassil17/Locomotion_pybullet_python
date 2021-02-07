import os
import numpy as np
import pybullet as p
import pybullet_data
from sim_env import SimEnv
from sim_robot import SimRobot
from robot_tracking_controller import Controller_robot
import math
from datetime import datetime

import time

urdfPath = os.getcwd() + '/talos_description/urdf/talos_lower_body_mesh_updated.urdf'
# urdfPath = os.getcwd() + '/talos_description/urdf/talos_full_v2_mesh_updated.urdf'
initial_pose = np.zeros(12)
initial_pose[2] =  -0.2
initial_pose[3] =  0.4
initial_pose[4] =  -0.2
initial_pose[8] =  -0.2
initial_pose[9] =  0.4
initial_pose[10] =  -0.2
t_homing = 0.5

sim_rate = 100
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)
# kinematic_test ==True is used when lifting the robot(kinematic test/joint tracking controller test),  kinematic_test ==False in walking simulation
Talos = SimRobot(urdfFileName=urdfPath,
                 basePosition=[-0.0, 0, 1.1],
                 baseRPY=[0, 0, 0],
                 jointPositions=initial_pose,
                 useFixedBase=False)

TalosId = Talos.id
# p.resetBasePositionAndOrientation(TalosId, [0, 0, 1.05], [0, 0, 0, 1])

# bad, get it from name! TalosEndEffectorIndex_l = 18
TalosEndEffectorIndex_l =6
TalosEndEffectorIndex_r =13
numJoints = p.getNumJoints(TalosId)
actuate_joint_indexes = Talos.getActuatedJointIndexes()
# joint damping coefficents
jd = [0.1, 0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]

p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0

useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
# trailDuration is duration (in seconds) after debug lines will be removed automatically
# use 0 for no-removal
trailDuration = 15

controler = Controller_robot(id=TalosId)


def getJointRanges(bodyId, Talos):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    q_current,lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], [], []

    actuate_joint_indexes = Talos.getActuatedJointIndexes()

    for i in range(len(actuate_joint_indexes)):
        j = actuate_joint_indexes[i]
        jointInfo = p.getJointInfo(bodyId, j)
        q = p.getJointState(bodyId, j)[0]

        ll, ul = jointInfo[8:10]
        jr = ul - ll

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(bodyId, j)[0]

        lowerLimits.append(ll)
        upperLimits.append(ul)
        jointRanges.append(jr)
        restPoses.append(rp)
        q_current.append(q)

    return q_current, lowerLimits, upperLimits, jointRanges, restPoses



def accurateIK(TalosId, endEffectorId, targetPosition1,targetPosition2, orn1,orn2, lowerLimits,upperLimits,jointRanges,restPoses,usingnullspace=False,using_orn=False,maxIter=10, threshold=1e-5):

    closeEnough = False
    iter = 0
    dist2 = 1e30

    endEffectorIdx = endEffectorId+7

    while (not closeEnough and iter < maxIter):
        if usingnullspace:
            if using_orn:

                ### right leg
                jointPosesx = p.calculateInverseKinematics(bodyUniqueId=TalosId, endEffectorLinkIndex=endEffectorIdx,
                                                          targetPosition=targetPosition2, targetOrientation=orn2,
                                                          lowerLimits=lowerLimits,
                                                          upperLimits=upperLimits, jointRanges=jointRanges,
                                                           restPoses=restPoses)
                ### left leg
                jointPosesx0 = p.calculateInverseKinematics(bodyUniqueId=TalosId, endEffectorLinkIndex=endEffectorId,
                                                          targetPosition=targetPosition1, targetOrientation=orn1,
                                                          lowerLimits=lowerLimits,
                                                          upperLimits=upperLimits, jointRanges=jointRanges,
                                                          restPoses=restPoses)
            else:
                ### right leg
                jointPosesx = p.calculateInverseKinematics(bodyUniqueId=TalosId, endEffectorLinkIndex=endEffectorIdx,
                                                          targetPosition=targetPosition2,
                                                          lowerLimits=lowerLimits,
                                                          upperLimits=upperLimits, jointRanges=jointRanges, restPoses=restPoses)

                jointPosesx0 = p.calculateInverseKinematics(bodyUniqueId=TalosId, endEffectorLinkIndex=endEffectorId,
                                                          targetPosition=targetPosition1,
                                                          lowerLimits=lowerLimits,
                                                          upperLimits=upperLimits, jointRanges=jointRanges, restPoses=restPoses)
        else:
            if using_orn:
                ### right leg
                jointPosesx = p.calculateInverseKinematics(bodyUniqueId=TalosId,
                                                          endEffectorLinkIndex=endEffectorIdx,
                                                          targetPosition=targetPosition2,
                                                          targetOrientation=orn2)
                ### left leg
                jointPosesx0 = p.calculateInverseKinematics(bodyUniqueId=TalosId,
                                                          endEffectorLinkIndex=endEffectorId,
                                                          targetPosition=targetPosition1,
                                                          targetOrientation=orn1)

            else:
                ### right leg
                jointPosesx = p.calculateInverseKinematics(bodyUniqueId=TalosId,
                                                          endEffectorLinkIndex=endEffectorIdx,
                                                          targetPosition=targetPosition2)

                jointPosesx0 = p.calculateInverseKinematics(bodyUniqueId=TalosId,
                                                          endEffectorLinkIndex=endEffectorId,
                                                          targetPosition=targetPosition1)

        jointPosesxx = np.array(jointPosesx)
        jointPosesx0x = np.array(jointPosesx0)

        jointPosesxxxxx = np.zeros(12)
        jointPosesxxxxx[0:6] = jointPosesx0x[0:6]
        jointPosesxxxxx[6:12] = jointPosesxx[6:12]
        jointPosesx1 = tuple(jointPosesxxxxx)

        joint_ref = list(jointPosesx1)
        # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for j in range(len(joint_ref)):
            if j < 6:
                p.resetJointState(TalosId, j, joint_ref[j])
            else:
                p.resetJointState(TalosId, j + 1, joint_ref[j])

        ls = p.getLinkState(TalosId, endEffectorId)
        newPos = ls[4]
        diff = [targetPosition1[0] - newPos[0], targetPosition1[1] - newPos[1], targetPosition1[2] - newPos[2]]
        dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
        # print("dist2=", dist2)
        ls1= p.getLinkState(TalosId, endEffectorIdx)
        newPos1 = ls1[4]
        diff1 = [targetPosition2[0] - newPos1[0], targetPosition2[1] - newPos1[1], targetPosition2[2] - newPos1[2]]
        dist21 = np.sqrt((diff1[0] * diff1[0] + diff1[1] * diff1[1] + diff1[2] * diff1[2]))

        closeEnough = ((dist21) < threshold)
        iter = iter + 1

    return jointPosesx1


lsx_ori = []

cid = p.createConstraint(TalosId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])

while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
        print(t)
    else:
        t = t + dt
        time.sleep(dt)

    q_current, lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(TalosId,Talos)

    lsx = p.getBasePositionAndOrientation(TalosId)[0]


    ### initial pose
    if t<t_homing:
        lsx_ori = p.getBasePositionAndOrientation(TalosId)[0]
        Talos.setActuatedJointPositions(initial_pose)
        left_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=6)[4]
        right_sole_urdf_position = p.getLinkState(bodyUniqueId=TalosId, linkIndex=13)[4]
        print("left_sole_urdf_position:",left_sole_urdf_position)
        print("right_sole_urdf_position:", right_sole_urdf_position)
    else:
        lsx1 = p.getBasePositionAndOrientation(TalosId)[0]
        # p.resetBasePositionAndOrientation(TalosId, [1, 0, 1.15], [0, 0, 0, 1])
        pos_base = [lsx_ori[0], lsx_ori[1]+(0.05*math.sin(t-t_homing)*2), lsx_ori[2]]
        p.changeConstraint(cid, pos_base, jointChildFrameOrientation=[0,0,0,1], maxForce=100)

        # pos1 = [-0.008999665267765522, 0.08522006124258041, 0.0013045195955783129]
        # pos2 = [-0.009104462340474129, -0.0848843976855278, 0.0013257551472634077+abs(0.05*math.sin((t-t_homing)*2))]

        pos1 = [0.078999665267765522, 0.04692094773054123, 0.0013045195955783129]
        pos2 = [0.079104462340474129, -0.12388730049133301 - 0.05*(t-t_homing), 0.0013257551472634077+abs(0.05*math.sin((t-t_homing)/20))]
        rpy1 = [0,0,0]
        rpy2 = [0,0,0]
        #
        # det_l = np.array(pos1) + np.array(lsx1) -  np.array(lsx_ori)
        # det_r = np.array(pos2) + np.array(lsx1) -  np.array(lsx_ori)
        #
        orn1 = p.getQuaternionFromEuler(rpy1)
        orn2 = p.getQuaternionFromEuler(rpy2)

        baseposition = p.getBasePositionAndOrientation(TalosId)[0]

        jointPoses1 = accurateIK(TalosId, TalosEndEffectorIndex_l, pos1,pos2,orn1,orn2,lowerLimits,upperLimits, jointRanges, restPoses,usingnullspace=True,using_orn=True)


        # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        # for j in range(len(jointPoses1)):
        #     if j < 6:
        #         p.resetJointState(TalosId, j, jointPoses1[j])
        #         joint_angle_m.append(p.getJointState(TalosId,j)[0])
        #     else:
        #         # pass
        #         p.resetJointState(TalosId, j+1, jointPoses1[j])
        #         joint_angle_m.append(p.getJointState(TalosId,j+1)[0])

        Talos.setActuatedJointPositions(jointPoses1)

    p.stepSimulation()
    # sim_env.step()
    ls = p.getBasePositionAndOrientation(TalosId)[0]
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
        p.addUserDebugLine(prevPose1, ls, [1, 0, 0], 1, trailDuration)
    prevPose = lsx
    prevPose1 = ls
    hasPrevPose = 1