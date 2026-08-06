# def compute_barrier_values(self):
    #     px = (
    #         self.robot_x +
    #         self.l * math.cos(self.robot_theta)
    #     )
    #     py = (
    #         self.robot_y +
    #         self.l * math.sin(self.robot_theta)
    #     )
    #     barriers = []
    #     for rid, robot in self.other_robots.items():
    #         if robot["x"] is None:
    #             continue
    #         h = (
    #             (px - robot["x"])**2 +
    #             (py - robot["y"])**2 -
    #             self.safe_distance**2
    #         )
    #         barriers.append(
    #             (
    #                 rid,
    #                 h
    #             )
    #         )

    #     return barriers




        # def test_clf_cbf(self):
    #     if self.stick_x is None:
    #         return
    #     px = (
    #         self.robot_x +
    #         self.l * math.cos(self.robot_theta)
    #     )
    #     py = (
    #         self.robot_y +
    #         self.l * math.sin(self.robot_theta)
    #     )
    #     goal_x = self.stick_x
    #     goal_y = self.stick_y
    #     ex = goal_x - px
    #     ey = goal_y - py
 
    #     # Nominal CLF control
    #     ux = self.kp * ex
    #     uy = self.kp * ey
    #     barriers = self.compute_barrier_derivatives(
    #         ux,
    #         uy
    #     )
    #     self.get_logger().info(
    #         f"ux={ux:.3f} "
    #         f"uy={uy:.3f}"
    #     )
    #     return barriers




        # def grab_the_stick(self):
    #     # arm forward
    #     self.arm_controller(
    #         x=0.20,
    #         z=0.05
    #     )

    #     # close gripper
    #     goal = GripperControl.Goal()
    #     goal.target_state = 2
    #     goal.power = 0.5
    #     self.gripper_action_client.send_goal_async(
    #         goal
    #     )

    #     # lift arm
    #     self.arm_controller(
    #         x=0.20,
    #         z=0.15
    #     )

    #     # back up
    #     self.msg_vel.linear.x = -0.10
    #     self.msg_vel.angular.z = 0.0

    #     # arm down
    #     self.arm_controller(
    #         x=0.15,
    #         z=0.05
    #     )

    #     self.l = 0.50
    #     self.state = "HAVE_STICK"

    #V1
    # def motion_controller(self):
 
    #     px = self.robot_x + self.l * math.cos(self.robot_theta) # Good
    #     py = self.robot_y + self.l * math.sin(self.robot_theta) # Good

    #     if self.stick_x is None:
    #         return

    #     pickup_distance = 0.20 # Good

    #     # Force robot to come straight from -Y
    #     goal_x = self.stick_x  # Good
    #     goal_y = (
    #         self.stick_y
    #         - pickup_distance
    #     )  # Good

    #     ex = goal_x - px
    #     ey = goal_y - py

    #     distance = math.sqrt(ex**2 + ey**2)

    #     # self.get_logger().info(
    #     #     f'Robot=({self.robot_x:.2f},{self.robot_y:.2f}) '
    #     #     f'Stick=({self.stick_x:.2f},{self.stick_y:.2f}) '
    #     #     f'Goal=({goal_x:.2f},{goal_y:.2f})'
    #     # )

    #     # desired_theta = (
    #     #     self.stick_theta +
    #     #     math.pi / 2.0
    #     # )
        
    #     desired_theta = math.atan2(
    #         self.stick_y - self.robot_y,
    #         self.stick_x - self.robot_x
    #     )
    #     theta_error = (
    #         desired_theta -
    #         self.robot_theta
    #     )

    #     while theta_error > math.pi:
    #         theta_error -= 2.0 * math.pi

    #     while theta_error < -math.pi:
    #         theta_error += 2.0 * math.pi
            
    #     # Orientation Condition
    #     if distance < 0.20:
    #         self.msg_vel.linear.x = 0.0
    #         self.msg_vel.angular.z = 0.0
    #         self.state = "GRAB_STICK"
    #         # self.get_logger().info(
    #         #     f"STATE = {self.state}"
    #         # )
    #         return

    #     ux = self.kp * ex # Good
    #     uy = self.kp * ey # Good

    #     theta = self.robot_theta
        
    #     v = (
    #         ux * math.cos(theta) +
    #         uy * math.sin(theta)
    #     ) # Good

    #     omega = (
    #         -ux * math.sin(theta) +
    #         uy * math.cos(theta)
    #     ) / self.l  # Good

    #     omega += 0.8 * theta_error

    #     v = max(min(v, 0.5), -0.5)
    #     omega = max(min(omega, 1.5), -1.5)

    #     self.msg_vel.linear.x = v
    #     self.msg_vel.angular.z = omega
