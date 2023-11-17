import numpy as np

from .controller import Controller


class JointLimitsController(Controller):


    def __init__(
        self,
        robot_config,
        min_joint_angles,
        max_joint_angles,
        joint_angles_thresh,
        k_p,
        k_d,
        max_torque=None
    ):
        super().__init__(robot_config)
        
        self.max_joint_angles = max_joint_angles
        self.min_joint_angles = min_joint_angles
        self.joint_angles_thresh = joint_angles_thresh
        self.max_torque = max_torque
        
        self.k_p = k_p
        self.k_d = k_d
        
        if (
            self.max_joint_angles.shape[0] != robot_config.N_JOINTS
            or self.min_joint_angles.shape[0] != robot_config.N_JOINTS
        ):
            raise Exception("joint angles vector incorrect size")
        
        self.J = None
        self.M_safe = None
        self.M_inv = None
        self.xi = None

    def generate(self, q, dq):
        upper_danger_dist = self.max_joint_angles - (q + self.joint_angles_thresh) 
        lower_danger_dist = self.min_joint_angles - (q - self.joint_angles_thresh) 

        upper_danger_bool = upper_danger_dist <= 0
        lower_danger_bool = lower_danger_dist >= 0
        
        upper_limit_bool = (self.max_joint_angles - q) <= 0
        lower_limit_bool = (self.min_joint_angles - q) >= 0
                
        upper_danger_u = np.maximum(upper_danger_dist / self.joint_angles_thresh, -1) * self.k_p
        lower_danger_u = np.minimum(lower_danger_dist / self.joint_angles_thresh, 1)  * self.k_p

        upper_danger_u = upper_danger_bool * upper_danger_u 
        lower_danger_u = lower_danger_bool * lower_danger_u
        
        upper_danger_uvel = upper_danger_bool * dq *-self.k_d
        lower_danger_uvel = lower_danger_bool * dq * self.k_d
        
        aux_u = upper_danger_u + lower_danger_u + upper_danger_uvel + lower_danger_uvel
        
        upper_xi = 0.5 * (1 + np.sin((upper_danger_dist/self.joint_angles_thresh) * np.pi  - np.pi/2))
        lower_xi = 0.5 * (1 + np.sin((lower_danger_dist/self.joint_angles_thresh) * np.pi - np.pi/2))
        
        xi = upper_xi * upper_danger_bool + lower_xi * lower_danger_bool
        
        aux_u = aux_u * xi
        

        J =  np.diag(upper_limit_bool + lower_limit_bool) * 1
        J2 = np.diag(upper_danger_bool + lower_danger_bool) * 1
        
        M = self.robot_config.M(q)
        M_inv = np.linalg.inv(M)
        
        M_safe_inv = np.dot(J2, np.dot(M_inv, J2.T))
        M_safe = np.linalg.pinv(M_safe_inv)
        
        tau_safe = np.dot(np.dot(J2.T, M_safe), aux_u.T).flatten() 
        
        self.J = J
        self.M_safe = M_safe
        self.M_inv = M_inv
        self.xi = xi
        
        return tau_safe