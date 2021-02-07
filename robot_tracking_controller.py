import os
import numpy as np
import math
import pybullet


class Gait_Controller:
    def __init__(self, urbodx, id, verbose=True):
        self.robot = urbodx
        self.id = id
        self.mass = urbodx.getRobotMass()
        self.jointnumber = urbodx.getNumJoints()  #####note that the jointnumber equals to the linkname when the base_link is not count
        self.g = 9.8
        if verbose:
            print('*' * 100 + '\nPyBullet Controller Info ' + '\u2193 ' * 20 + '\n' + '*' * 100)
            print('robot ID:              ', id)
            xxx = self.robot.id
            print('urbodx.ID:              ', xxx)
            print('robot mass:              ', self.mass)
            print('*' * 100 + '\nPyBullet Controller Info ' + '\u2191 ' * 20 + '\n' + '*' * 100)

    def read_data_offline(self, FilePathName):
        Input = open(FilePathName, 'r')
        maximan_column = 3
        datax = []
        for i in range(0, maximan_column):
            try:
                a = Input.readline()
                ax = a.split()
                datax.append(ax)
            except:
                pass
        # data = map(map,[float,float,float,float,float,float,float,float,float,float], data)
        data = np.array(datax)
        print('data size', data.shape)

        return data

    def state_estimation(self, i, dt, support_flag, links_pos_prev, links_vel_prev, gcom_pre):
        gcom, right_sole_pos, left_sole_pos, base_pos, base_angle = self.cal_com_state()
        right_ankle_force, left_ankle_force = self.ankle_joint_pressure()
        links_pos, links_vel, links_acc = self.get_link_vel_vol(i, dt, links_pos_prev, links_vel_prev)
        gcop, support_flag, dcm_pos, com_vel = self.cal_cop(i, support_flag, links_pos, links_acc, right_ankle_force,
                                                            left_ankle_force, right_sole_pos, left_sole_pos, gcom_pre,
                                                            gcom, dt)

        return gcom, right_sole_pos, left_sole_pos, base_pos, base_angle, right_ankle_force, left_ankle_force, gcop, support_flag, dcm_pos, com_vel,links_pos, links_vel, links_acc

    ####### state feedback and recomputation#####################################
    ###### com state, foot location, baseposition and orientation===========================
    def cal_com_state(self):
        total_mass_moment = [0, 0, 0]
        com_pos = [0, 0, 0]

        for linkId in range(0, self.jointnumber):
            link_com_pos = pybullet.getLinkState(self.id, linkId)[0]
            total_mass_moment[0] += link_com_pos[0] * self.robot.getLinkMass(linkId)
            total_mass_moment[1] += link_com_pos[1] * self.robot.getLinkMass(linkId)
            total_mass_moment[2] += link_com_pos[2] * self.robot.getLinkMass(linkId)

        base_link_pos,base_orn = pybullet.getBasePositionAndOrientation(self.id)
        total_mass_moment[0] += base_link_pos[0] * self.robot.getLinkMass(-1)
        total_mass_moment[1] += base_link_pos[1] * self.robot.getLinkMass(-1)
        total_mass_moment[2] += base_link_pos[2] * self.robot.getLinkMass(-1)

        # total_mass_moment = [total_mass_momentx,total_mass_momenty,total_mass_momentz]
        com_pos[0] = total_mass_moment[0] / self.mass
        com_pos[1] = total_mass_moment[1] / self.mass
        com_pos[2] = total_mass_moment[2] / self.mass

        ###### footsole position: eliminating the offset
        right_sole_pos = list(pybullet.getLinkState(self.id, self.jointnumber - 1)[0])
        right_sole_pos[2] -= 0.0013
        left_sole_pos = list(pybullet.getLinkState(self.id, self.jointnumber - 8)[0])
        left_sole_pos[2] -= 0.00129


        base_angle = pybullet.getEulerFromQuaternion(base_orn)

        return com_pos, right_sole_pos, left_sole_pos, base_link_pos, base_angle

    ###### multilink model for ZMP and DCM calculation: kajita et al.  introduction to humanoid robots
    #### ankle joint force/torque sensor
    def ankle_joint_pressure(self):
        rig_legid = self.jointnumber - 2
        left_legid = self.jointnumber - 9
        right_leg_6_joint_info = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=rig_legid)[2]
        left_leg_6_joint_info = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=left_legid)[2]

        return right_leg_6_joint_info, left_leg_6_joint_info

    ##### getlink velocity and acceleration
    def get_link_vel_vol(self, i, dt, links_pos_pre, links_vel_pre):
        links_pos = np.zeros([self.jointnumber + 1, 3])
        links_vel = np.zeros([self.jointnumber + 1, 3])
        links_acc = np.zeros([self.jointnumber + 1, 3])
        for linkId in range(0, self.jointnumber):
            links_pos[linkId] = pybullet.getLinkState(self.id, linkId, computeLinkVelocity=1)[0]
            links_vel[linkId] = pybullet.getLinkState(self.id, linkId, computeLinkVelocity=1)[6]  #### link velocity

        links_pos[self.jointnumber] = pybullet.getBasePositionAndOrientation(self.id)[0]
        links_vel[self.jointnumber] = pybullet.getBaseVelocity(self.id)[0]

        if (i >= 1):
            links_acc = (links_vel - links_vel_pre) / dt

        return links_pos, links_vel, links_acc

    ##### ZMP/DCM calculation
    def cal_cop(self, i, support_flag, links_pos, links_acc, right_ankle_force, left_ankle_force, right_sole_pos,
                left_sole_pos, gcom_pre, gcom, dt):
        total_vetical_acce = 0.0
        total_vetical_accx = 0.0
        total_vetical_accy = 0.0
        total_forwar_accz = 0.0
        total_lateral_accz = 0.0

        cop_state = [0, 0, 0]
        com_vel = [0, 0, 0]
        dcm_pos = [0, 0, 0]

        if (i == 0):

            support_flag[i] = 0
            dcm_pos[0] = gcom[0]
            dcm_pos[1] = gcom[1]
            com_vel = [0, 0, 0]
        else:
            #### computing the pz:
            if (abs(right_ankle_force[2]) > 100):  #### right leg is touching the ground
                if (abs(left_ankle_force[2]) > 100):  #### left leg is touching the ground
                    support_flag[i] = 0
                    if (support_flag[
                        i - 1] <= 1):  ###### from right support switching to left support, taking the right support as the current height
                        cop_state[2] = right_sole_pos[2]
                        flagxxx = 1
                    else:
                        cop_state[2] = left_sole_pos[
                            2]  ###### from left support switching to right support, taking the left support as the current height
                        flagxxx = 2

                else:  ###right support
                    cop_state[2] = right_sole_pos[2]
                    support_flag[i] = 1  ### right support
                    flagxxx = 3
            else:
                if (abs(left_ankle_force[2])) > 100:  #### left leg is touching the ground
                    cop_state[2] = left_sole_pos[2]
                    support_flag[i] = 2  ### left support
                    flagxxx = 4
                else:
                    cop_state[2] = (right_sole_pos[2] + left_sole_pos[2]) / 2
                    support_flag[i] = 0
                    flagxxx = 5

            for linkId in range(0, self.jointnumber + 1):
                total_vetical_acce += (links_acc[linkId, 2] + self.g)
                total_vetical_accx += ((links_acc[linkId, 2] + self.g) * links_pos[linkId, 0])
                total_vetical_accy += ((links_acc[linkId, 2] + self.g) * links_pos[linkId, 1])
                total_forwar_accz += ((links_pos[linkId, 2] - cop_state[2]) * links_acc[linkId, 0])
                total_lateral_accz += ((links_pos[linkId, 2] - cop_state[2]) * links_acc[linkId, 1])

            cop_state[0] = (total_vetical_accx - total_forwar_accz) / total_vetical_acce
            cop_state[1] = (total_vetical_accy - total_lateral_accz) / total_vetical_acce

            com_vel[0] = (gcom[0] - gcom_pre[0]) / dt
            com_vel[1] = (gcom[1] - gcom_pre[1]) / dt
            com_vel[2] = (gcom[2] - gcom_pre[2]) / dt

            dcm_omega = np.sqrt(self.g / (gcom[2] - cop_state[2]))
            dcm_pos[0] = gcom[0] + 1.0 / dcm_omega * com_vel[0]
            dcm_pos[1] = gcom[1] + 1.0 / dcm_omega * com_vel[1]
            dcm_pos[2] = cop_state[2]

        return cop_state, support_flag, dcm_pos, com_vel

    ##### controller-block#############################
    ###### IK-based controller
    ##### CoM/Body PD controller
    def CoM_Body_pd(self,dt,com_ref_det,com_feedback_det,com_ref_det_pre,com_feedback_det_pre,angle_ref_det,angle_feedback_det,angle_ref_det_pre,angle_feedback_det_pre):
        det_com = [0,0,0]
        det_body_angle = [0, 0, 0]
        det_com[0] = 0.2* (com_ref_det[0]-com_feedback_det[0]) +  0.00001* (com_ref_det[0]-com_feedback_det[0] - (com_ref_det_pre[0]-com_feedback_det_pre[0]))/dt
        det_com[1] = 0.05* (com_ref_det[1]-com_feedback_det[1]) +  0.00001* (com_ref_det[1]-com_feedback_det[1] - (com_ref_det_pre[1]-com_feedback_det_pre[1]))/dt
        det_com[2] = 0.1* (com_ref_det[2]-com_feedback_det[2]) +  0.00001* (com_ref_det[2]-com_feedback_det[2] - (com_ref_det_pre[2]-com_feedback_det_pre[2]))/dt
        det_body_angle[0] = 0.01* (angle_ref_det[0]-angle_feedback_det[0]) +  0.00001* (angle_ref_det[0]-angle_feedback_det[0] - (angle_ref_det_pre[0]-angle_feedback_det_pre[0]))/dt
        det_body_angle[1] = 0.1* (angle_ref_det[1]-angle_feedback_det[1]) +  0.00001* (angle_ref_det[1]-angle_feedback_det[1] - (angle_ref_det_pre[1]-angle_feedback_det_pre[1]))/dt
        det_body_angle[2] = 0.01* (angle_ref_det[2]-angle_feedback_det[2]) +  0.00001* (angle_ref_det[2]-angle_feedback_det[2] - (angle_ref_det_pre[2]-angle_feedback_det_pre[2]))/dt

        return det_com, det_body_angle

    ####### ZMP preview controller

    #####  whole-body controller
    def com_horizontal_admittance(self):
        det_com = [0,0,0]
        return det_com






    ####### Rotation matrix generated by the  rpy angle
    def RotMatrixfromEuler(self, xyz):
        x_angle = xyz[0]
        y_angle = xyz[1]
        z_angle = xyz[2]
        Rrpy = np.array([[math.cos(y_angle) * math.cos(z_angle), math.cos(z_angle) * math.sin(x_angle) * \
                          math.sin(y_angle) - math.cos(x_angle) * math.sin(z_angle),
                          math.sin(x_angle) * math.sin(z_angle) + \
                          math.cos(x_angle) * math.cos(z_angle) * math.sin(y_angle)],
                         [math.cos(y_angle) * math.sin(z_angle), math.cos(x_angle) * math.cos(z_angle) + \
                          math.sin(x_angle) * math.sin(y_angle) * math.sin(z_angle),
                          math.cos(x_angle) * math.sin(y_angle) * math.sin(z_angle) \
                          - math.cos(z_angle) * math.sin(x_angle)],
                         [-math.sin(y_angle), math.cos(y_angle) * math.sin(x_angle),
                          math.cos(x_angle) * math.cos(y_angle)]])
        # qua_base = pybullet.getQuaternionFromEuler(xyz)
        # Rrpy = pybullet.getMatrixFromQuaternion(qua_base)

        return Rrpy
