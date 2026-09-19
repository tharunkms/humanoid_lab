nit_solver()
        self.mpc.init_constraint_args()
 
        # 3. Setup the initial run state and logging matrices
        self.mpc.prepare_step(self.current_state)
        self.mpc.init_sim_params()
 
        # 4. ROS Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(int(1.0 / self.mpc.step_horizon))
 
    def run(self):
        rospy.loginfo("Starting Modular MPC Control Loop...")
 
        while not rospy.is_shutdown() and not self.mpc.mpc_completed:
 
            # --- 1. CALCULATE OPTIMAL VELOCITY ---
            # Using the new simulation-parameter-aware step function
            u = self.mpc.step_with_sim_params(self.current_state, self.target_state)
 
            v_cmd = float(u[0, 0])
            w_cmd = float(u[1, 0])
 
            # --- 2. PUBLISH TO ROS ---
            twist_msg = Twist()
            twist_msg.linear.x = v_cmd
            twist_msg.angular.z = w_cmd
            self.cmd_pub.publish(twist_msg)
 
            # --- 3. ADVANCE THE MATH MODEL ---
            # Using your new clean integration method
            self.current_state = self.mpc.simulate_step_shift(u, self.current_state)
 
            # Sleep to lock the loop timing to the step_horizon
            self.rate.sleep()
 
        # Stop the physical/simulated robot once target is reached
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Target Reached. Compiling Animation...")
 
 
if __name__ == '__main__':
    try:
        MPCROSController().run()
    except rospy.ROSInterruptException:
        pass