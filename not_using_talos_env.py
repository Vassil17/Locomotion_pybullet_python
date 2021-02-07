import os
import numpy as np
import pybullet
from sim_env import SimEnv
from sim_robot import SimRobot
from robot_tracking_controller import Controller_robot
import math
from datetime import datetime

import time

if __name__ == '__main__':

    sim_rate = 100
    g = 9.81
    # urdfPath = os.getcwd() + '/talos_description/urdf/talos_lower_body_mesh_updated.urdf'
    urdfPath = os.getcwd() + '/talos_description/urdf/talos_full_v2_mesh_updated.urdf'
    # trailDuration is duration (in seconds) after debug lines will be removed automatically
    # use 0 for no-removal
    trailDuration = 5

    # lower limits for null space
    ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # upper limits for null space
    ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # restposes for null space
    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ikSolver = 0

    prevPose = [0,0,0]
    pos = [0,0,0]
    t = 0.
    # homing_config = np.zeros(robot.getNumActuatedJoints())

    sim_env = SimEnv(sim_rate=sim_rate)
    # kinematic_test ==True is used when lifting the robot(kinematic test/joint tracking controller test),  kinematic_test ==False in walking simulation
    robot = SimRobot(urdfFileName=urdfPath,
                     basePosition=[-0.0, 0, 1.1],
                     baseRPY=[0, 0, 0],
                     useFixedBase=False)

    sim_env.reset()
    pybullet.stepSimulation()

    # controler = Controller_robot(id=robot.id)
    # comfile = r'/media/jiatao/23c76e85-0a74-40fd-afd2-fd3762658c39/jiatao/airs_uoe/nao_experiment_code/nao_20160504straight5_mod_2s_100_40_angle.txt'
    # com_offline = controler.read_data_offline(FilePathName=comfile)
    # Left_leg_jointPositions = robot.calculateInverseKinematics('left_sole_link', [0.15, 0.1, 0.3], [0, 0*math.pi/180, 0])
    # Right_leg_jointPositions = robot.calculateInverseKinematics('right_sole_link', [0.0, -0.1, 0.2], [0, 0, 0])
    #
    # joint_ref = controler.joint_rearragement(Left_leg_joint=Left_leg_jointPositions,Right_leg_joint=Right_leg_jointPositions)
    left_sole_link_position = pybullet.getLinkState(bodyUniqueId=robot.id, linkIndex=5)[0]
    right_sole_link_position = pybullet.getLinkState(bodyUniqueId=robot.id, linkIndex=12)[0]
    basePosition = robot.getBaseCoMPosition()
    basklink_position = robot.getBaseLinkPosition()
    left_sole_urdf_position = pybullet.getLinkState(bodyUniqueId=robot.id, linkIndex=5)[4]
    right_sole_urdf_position = pybullet.getLinkState(bodyUniqueId=robot.id, linkIndex=12)[4]



    useRealTimeSimulation =0
    use_simulation = 1
    count = 0
    while 1:
        1
    #     if (useRealTimeSimulation):
    #         dt = datetime.now()
    #         t = (dt.second / 60.) * 2. * math.pi
    #     else:
    #         t = t + 0.01
    #
    #     # sim_env.step()
    #     prevPose = robot.getBaseCoMPosition()
    #     robot.setfixedbase(kinematic_test=True)
    #
    #     # joint command test:
    #     if (use_simulation and useRealTimeSimulation == 0):
    #         pybullet.stepSimulation()
    #
    #     if (count==151) or (count==151) :
    #         robotid = robot.id
    #         print('robot id:',robotid)
    #         left_sole_urdf_position = [0.15, 0.1, 0.3+0.01*math.sin(t)]
    #         left_sole_urdf_angle = [0,0,0]
    #         left_sole_linkid = 6
    #         right_sole_urdf_position = [0.0, -0.1+0.001*math.cos(t), 0.2]
    #         right_sole_urdf_angle = [0,0,0]
    #         right_sole_linkid = 13
    #         # Left_leg_jointPositions = robot.calculateInverseKinematics('left_sole_link', left_sole_urdf_position,
    #         #                                                            [0, 0, 0])
    #         # arr1 = np.array(Left_leg_jointPositions,dtype=float)
    #         #
    #         # print('left_sole_urdf_position:', left_sole_urdf_position)
    #         # print('right_sole_urdf_position:', right_sole_urdf_position)
    #         #
    #         # Right_leg_jointPositions = robot.calculateInverseKinematics('right_sole_link', right_sole_urdf_position,
    #         #                                                             [0, 0, 0])
    #         # arr2 = np.array(Right_leg_jointPositions,dtype=float)
    #
    #         Left_leg_jointPositions = pybullet.calculateInverseKinematics(robotid,
    #                                                                       left_sole_linkid,
    #                                                                       left_sole_urdf_position,
    #                                                                       left_sole_urdf_angle,
    #                                                                       jointDamping=jd,
    #                                                                       solver=ikSolver,
    #                                                                       maxNumIterations=100,
    #                                                                       residualThreshold=.001
    #                                                                        )
    #         Right_leg_jointPositions = pybullet.calculateInverseKinematics(robotid,
    #                                                                        right_sole_linkid,
    #                                                                        right_sole_urdf_position,
    #                                                                        right_sole_urdf_angle,
    #                                                                        jointDamping=jd,
    #                                                                        solver=ikSolver,
    #                                                                        maxNumIterations=100,
    #                                                                        residualThreshold=.001)
    #         # print('Left_leg_jointPositions:', np.array(Left_leg_jointPositions) - np.array(Left_leg_jointPositions1))
    #         # print('Right_leg_jointPositions:', np.array(Right_leg_jointPositions)-np.array(Right_leg_jointPositions1))
    #
    #         joint_ref = controler.joint_rearrangement(Left_leg_joint=Left_leg_jointPositions,
    #                                                  Right_leg_joint=Right_leg_jointPositions)
    #
    #
    #         if use_simulation:
    #             robot.setActuatedJointPositions(joint_ref)
    #         else:
    #             for i in range(len(Left_leg_jointPositions)):
    #                 if i < 6:
    #                     pybullet.resetJointState(robot.id, i, joint_ref[i])
    #                     # pybullet.setJointMotorControl2(bodyIndex=robot.id,
    #                 #                                jointIndex=i,
    #                 #                                controlMode=pybullet.POSITION_CONTROL,
    #                 #                                targetPosition=joint_ref[i],
    #                 #                                targetVelocity=0,
    #                 #                                force=500,
    #                 #                                positionGain=1,
    #                 #                                velocityGain=1)
    #                 else:
    #                     pybullet.resetJointState(robot.id, i + 1, joint_ref[i])
    #                     # pybullet.setJointMotorControl2(bodyIndex=robot.id,
    #                     #                                jointIndex=i+1,
    #                     #                                controlMode=pybullet.POSITION_CONTROL,
    #                     #                                targetPosition=joint_ref[i],
    #                     #                                targetVelocity=0,
    #                     #                                force=500,
    #                     #                                positionGain=1,
    #                     #                                velocityGain=1)
    #
    #     time.sleep(0.01)
    #     # sim_env.debug()
    #     pos = robot.getBaseCoMPosition()
    #     pybullet.addUserDebugLine(prevPose, pos, [1, 0, 0.3], 1, trailDuration)
    #     count+=1
