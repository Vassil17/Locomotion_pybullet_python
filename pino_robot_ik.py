from __future__ import print_function

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve

#### Jacobian based Ik
class CLIK:
    def __init__(self, robot, oMdes, index,Freebase):
        self.oMdes = oMdes
        self.index = index
        self.Freebase = Freebase
        self.robot = robot


    def ik_Jacobian(self, q, Freebase, eps= 1e-4,IT_MAX = 1000,DT = 1e-1,damp = 1e-6):
        i=0
        while True:
            pin.forwardKinematics(self.robot.model,self.robot.data,q)
            dMi = self.oMdes.actInv(self.robot.data.oMi[self.index])
            err = pin.log(dMi).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            J = pin.computeJointJacobian(self.robot.model,self.robot.data,q,self.index)
            if Freebase:
                J[0:6,0:7] = np.zeros((6,7))
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            if Freebase:
                v[0:7] = np.zeros([7])
            q = pin.integrate(self.robot.model,q,v*DT)
            if Freebase:
                q[0:7] = np.zeros([7])

            # if not i % 10:
            #     print('%d: error = %s' % (i, err.T))
            i += 1
        # print('\nresult: %s' % q[7:13])
        # print('\nfinal error: %s' % err.T)
        # pin.forwardKinematics(robot.model, robot.data, q)
        # p = robot.data.oMi[self.index].translation
        # print(ft ankle joint result-Jaco:", p)
        J = pin.computeJointJacobian(self.robot.model, self.robot.data, q, self.index)
        # print("*************************************************************************")
        return q, J

# def position_error(self, q):
#     # ### considering the orn and pos: expected status
#     no = log(self.oMdes)
#     log_oMdes_vec = no.vector
#
#     #### calculating the robot status:
#     pin.forwardKinematics(self.robot.model, self.robot.data, q)
#
#     p = self.robot.data.oMi[self.index]
#     nv = log(p)
#     log_po_vec = nv.vector
#
#     err1 = np.sqrt((log_oMdes_vec[0] - log_po_vec[0]) ** 2 + (log_oMdes_vec[1] - log_po_vec[1]) ** 2 + (
#             log_oMdes_vec[2] - log_po_vec[2]) ** 2 + (log_oMdes_vec[3] - log_po_vec[3]) ** 2 + (
#                            log_oMdes_vec[4] - log_po_vec[4]) ** 2 + (log_oMdes_vec[5] - log_po_vec[5]) ** 2)
#     err2 = np.sqrt((q[0]) ** 2 + (q[1]) ** 2 + (q[2]) ** 2 + (q[3]) ** 2 + (q[4]) ** 2 + (q[5]) ** 2 + (
#         q[6]) ** 2)  ####base joint
#     err = 100 * err1 + 20 * err2
#     return err
#
#
#
#
#
#
# def fbgs_opt(self,q):
#
#     xopt_bfgs = fmin_bfgs(position_error, q)
#     # print('*** Xopt in BFGS =', xopt_bfgs[7:13])
#     return xopt_bfgs

