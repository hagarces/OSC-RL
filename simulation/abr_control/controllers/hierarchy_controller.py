import numpy as np

from .controller import Controller


class HierarchyController(Controller):


    def __init__(
        self,
        robot_config,
        safe_controller,
        max_torque,
        min_torque,
        op_space_controller = None,
        null_controllers=None,
    ):
        super().__init__(robot_config)
        
        self.safe_controller = safe_controller
        self.op_space_controller = op_space_controller
        self.null_controllers = null_controllers

        self.max_torque = max_torque
        self.min_torque = min_torque
        
        self.test_safe = 0
        self.test_osc = 0
        self.test_safe_n_osc = 0
        self.test_osc_n_null = 0
        
        self.u_safe = None
        self.u_osc_ = None
        self.no_null = None

    def generate(self, q, dq, target, target_velocity=None, ref_frame="EE",
                 xyz_offset=None):

        u_safe = self.safe_controller.generate(q, dq) 
        
        J1 = self.safe_controller.J
        M_safe = self.safe_controller.M_safe
        M_inv = self.safe_controller.M_inv
        xi = self.safe_controller.xi

        N_1 = np.eye(J1.shape[0]) - np.dot(np.dot(J1, M_safe), np.dot(J1, M_inv))
        if self.op_space_controller:
            u_osc_ = self.op_space_controller.generate(q, dq, target, target_velocity, ref_frame,
                     xyz_offset)
            
            J2 = self.op_space_controller.J
            
            u_osc = np.dot(N_1, u_osc_)

            u_null = 0
            if self.null_controllers is not None:
                for null_controller in self.null_controllers:
    
                    u_null_ = null_controller.generate(q, dq,
                                                      target_velocity=np.array([0] * self.robot_config.DOF))
                    
                    u_null += u_null_
            
                J_aux = np.dot(J2, N_1)
                M_x_safe = np.linalg.pinv(np.dot(J_aux, np.dot(M_inv, J_aux.T)))
                J_aux_inv = np.dot(np.dot(M_inv, J_aux.T), M_x_safe)
                N_2 = N_1 - np.dot(J_aux.T, J_aux_inv.T) 
                
                u_null = np.dot(N_2, u_null)
        else:
            return u_safe
            
        if self.test_safe:
            self.u_safe = np.clip(u_safe, self.min_torque, self.max_torque)
            return self.u_safe
        
        if self.test_osc:
            self.u_osc_ = np.clip(u_osc_, self.min_torque, self.max_torque)
            return self.u_osc_
        
        if self.test_safe_n_osc:
            self.safe_n_osc = np.clip(u_safe * xi + (1 - xi) * u_osc, self.min_torque, self.max_torque)
            return self.safe_n_osc
        
        if self.test_osc_n_null:
            self.osc_n_null =  np.clip(u_osc + u_null, self.min_torque, self.max_torque)
            return self.osc_n_null

        u = u_safe * xi + (1 - xi) * ( u_osc + u_null)
        
        u = np.clip(u, self.min_torque, self.max_torque)
        return u

