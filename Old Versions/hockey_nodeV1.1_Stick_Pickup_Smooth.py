import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

#from rosidl_runtime_py import message_to_yaml
from geometry_msgs.msg import Twist, PoseStamped, Point
from robomaster_msgs.action import GripperControl

ROBOT_ID = 10

class AlgorithmNode(Node):
    def __init__(self, robot_id):
        node_name = f'algorithm_node_robot{robot_id}'
        super().__init__(node_name)

        # Topic & Action Names
        pose_topic = f'/vrpn_mocap/dji_robot_{robot_id}/pose'
        cmd_vel_topic = f'/robot{robot_id}/cmd_vel'
        target_arm_topic = f'/robot{robot_id}/target_arm_position'  # CHANGED: This is now ABSOLUTE position
        gripper_action = f'/robot{robot_id}/gripper'

        # Constants
        self.CONTROL_LOOP_FREQUENCIES = 50  # Hz
        
        # State Tracking
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        #Stick state
        self.stick_x = None
        self.stick_y = None
        self.stick_theta = None

        # Puck
        self.puck_x = None
        self.puck_y = None
        
        # Goal
        self.goal_x = None
        self.goal_y = None
        
        # Other Robots
        self.other_robots = {}
        for rid in range(1, 11):
            if rid == ROBOT_ID:
                continue
            self.other_robots[rid] = {
                "x": None,
                "y": None,
                "theta": None
            }
        
        self.debug_counter = 0    # Debugging counter for logging
        self.safe_distance = 0.35 # Safe distance for CBF (Collision Avoidance)
        self.gamma = 1.0          

        # Approximate linearization parameter
        self.kp = 1.2 #0.85            # Navigation Aggressiveness
        self.l = 0.30            # Distance from robot center to virtual point P 50
        self.state = "GET_CHECKPOINT" # State tracking for the state machine
        self.STEER_GAIN_L = 0.5
        self.prev_v = 0.0
        self.prev_omega = 0.0

        # Timer
        self.action_step = 0
        self.action_start_time = None

        # Subscribers If Issue Check for Subscribers
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.sub_pose = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.sub_pose_callback,
            best_effort_qos
        )
        self.sub_stick = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/hockey_sticks_4/pose',   #ISSUE Change for Simulation and robot 
            self.stick_callback,
            best_effort_qos
        )
        self.sub_puck = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/hockey_puck_violet/pose',
            self.puck_callback,
            best_effort_qos
        )
        self.sub_goal = self.create_subscription(
            PoseStamped,
            '/vrpn_mocap/hockey_goal_1/pose',
            self.goal_callback,
            best_effort_qos
        )

        # Subscribe to all other robots
        self.robot_subscribers = []
        for rid in range(1, 11):
            if rid == ROBOT_ID:
                continue
            topic = (
                f'/vrpn_mocap/dji_robot_{rid}/pose'
            )
            sub = self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, rid=rid:
                    self.other_robot_callback(msg, rid),
                best_effort_qos
            )
            self.robot_subscribers.append(sub)

        # Publishers
        self.msg_vel = Twist()
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 20)
        self.msg_target_arm = Point()                                            # NOTE: Target_arm_position expects geometry_msgs/Point
        self.pub_target_arm = self.create_publisher(Point, target_arm_topic, 20)

        #Initialize the arm to Start position
        self.msg_target_arm.x = 0.10
        self.msg_target_arm.y = 0.0
        self.msg_target_arm.z = 0.05
        self.pub_target_arm.publish(
            self.msg_target_arm
        )

        # Action Clients
        self._action_group = ReentrantCallbackGroup()
        self.gripper_action_client = ActionClient(
            self,
            GripperControl,
            gripper_action,
            callback_group=self._action_group
        )

        # Wait for the action server to be online before starting
        self.get_logger().info(f'Waiting for action server: {gripper_action}...')
        self.gripper_action_client.wait_for_server()
        self.get_logger().info('Gripper action server found! Starting control loop.')
        
        # Open gripper
        goal = GripperControl.Goal()
        goal.target_state = 1
        goal.power = 0.5
        self.gripper_action_client.send_goal_async(
            goal
        )

        # Control loop Timer 
        self.timer = self.create_timer(1.0 / self.CONTROL_LOOP_FREQUENCIES, self.timer_callback)

#----------------------------Callback & derivatives---------------------------------#
    def sub_pose_callback(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.robot_theta = 2.0 * math.atan2(qz, qw)
        # self.get_logger().info(
        #     f'x={self.robot_x:.2f}, '
        #     f'y={self.robot_y:.2f}, '
        #     f'theta={self.robot_theta:.2f}'
        # )
        
    def stick_callback(self, msg):
        self.stick_x = msg.pose.position.x
        self.stick_y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.stick_theta = 2.0 * math.atan2(qz, qw)
        # self.get_logger().info(
        #    f'Stick: '
        #    f'x={self.stick_x:.2f}, '
        #    f'y={self.stick_y:.2f}, '
        #    f'theta={math.degrees(self.stick_theta):.1f}'
        # )
        
    def puck_callback(self, msg):
        self.puck_x = msg.pose.position.x
        self.puck_y = msg.pose.position.y

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y   

    def other_robot_callback(self, msg, robot_id):
        self.other_robots[robot_id]["x"] = (msg.pose.position.x)
        self.other_robots[robot_id]["y"] = (msg.pose.position.y)
        qz = (msg.pose.orientation.z)
        qw = (msg.pose.orientation.w)
        self.other_robots[robot_id]["theta"] = (2.0 * math.atan2(qz, qw))

    def get_nearby_robots(self):
        nearby_robots = []
        for rid, robot in self.other_robots.items():
            if robot["x"] is None:
                continue
            distance = math.sqrt( (self.robot_x - robot["x"])**2 + (self.robot_y - robot["y"])**2)
            if distance < 1.5:
                nearby_robots.append((rid, robot["x"], robot["y"], distance))

        return nearby_robots

    def compute_barrier_derivatives(self, ux, uy):
        px = (self.robot_x + self.l * math.cos(self.robot_theta))
        py = (self.robot_y + self.l * math.sin(self.robot_theta))

        barriers = []
        nearby_robots = self.get_nearby_robots()

        for rid, x_obs, y_obs, distance in nearby_robots:
            if x_obs is None:
                continue
            dx = (px - x_obs)
            dy = (py - y_obs)
            h = (dx**2 + dy**2 - self.safe_distance**2)
            h_dot = (2.0 * dx * ux + 2.0 * dy * uy)
            cbf_value = (h_dot + self.gamma * h)
            barriers.append((rid, h, h_dot, cbf_value))
        return barriers

    def solve_clf_cbf_qp(self, ux_nom, uy_nom):
        ux = ux_nom
        uy = uy_nom

        px = (
            self.robot_x + self.l * math.cos(self.robot_theta))

        py = (self.robot_y + self.l * math.sin(self.robot_theta))

        nearby_robots = (self.get_nearby_robots())

        for rid, x_obs, y_obs, distance in nearby_robots:
            dx = px - x_obs
            dy = py - y_obs
            h = (
                dx**2 +
                dy**2 -
                self.safe_distance**2
            )
            a1 = 2.0 * dx
            a2 = 2.0 * dy
            cbf = (
                a1 * ux +
                a2 * uy +
                self.gamma * h
            )
            if cbf < 0.0:
                norm_sq = (
                    a1*a1 +
                    a2*a2
                )
                if norm_sq > 1e-6:
                    correction = (
                        -cbf /
                        norm_sq
                    )
                    ux += (
                        correction * a1
                    )
                    self.get_logger().info(
                        f"CBF ACTIVE R{rid}"
                    )
                    uy += (
                        correction * a2
                    )
                    # Canceling only the radial (into-obstacle) component is
                    # the correct minimal safety correction, but when the
                    # target sits almost directly through the obstacle it
                    # leaves nearly zero residual velocity -- the robot
                    # crawls to a stop instead of curving around, only
                    # regaining speed once drift happens to open an angle.
                    # A small nudge purely along the TANGENT (perpendicular
                    # to a1,a2) doesn't change h_dot at all here, so it can't
                    # violate the constraint we just enforced -- it just
                    # gives the robot somewhere safe to actively go instead
                    # of stalling. Side is picked from the ORIGINAL nominal
                    # (not the post-correction ux,uy, which can be near zero
                    # and noisy) so it doesn't flip-flop tick to tick.
                    #
                    # The nudge's MAGNITUDE ramps continuously with how hard
                    # the constraint is binding (proportional to the radial
                    # correction just applied), instead of being a fixed
                    # value that switches fully on/off right at cbf=0 -- a
                    # discontinuity that showed up as overturning followed
                    # immediately by an opposite overcorrection every time
                    # the robot grazed the edge of the safe zone.
                    tangent_x = -a2
                    tangent_y = a1
                    tangent_norm = math.sqrt(
                        tangent_x**2 + tangent_y**2
                    )
                    if tangent_norm > 1e-6:
                        tangent_x /= tangent_norm
                        tangent_y /= tangent_norm
                        side = ux_nom * tangent_x + uy_nom * tangent_y
                        sign = 1.0 if side >= 0.0 else -1.0
                        radial_correction_mag = -cbf / tangent_norm
                        nudge = min(radial_correction_mag, 0.3)
                        ux += sign * nudge * tangent_x
                        uy += sign * nudge * tangent_y
        return ux, uy

    def apply_velocity(self, v, omega):
        """Low-pass filters v/omega against the previous tick's command
        before publishing. Each control method recomputes v/omega from
        scratch every tick (nominal control law + whatever the CBF
        correction does that instant); when the CBF is continuously active
        for a long stretch (e.g. skimming past a nearby robot the whole
        approach), that per-tick recomputation has no memory of the
        previous command, so any small overshoot in one direction gets
        corrected, then overshoots the other way, tick after tick -- a
        zigzag. Blending with the previous command damps that out without
        changing the underlying control law."""
        smoothing = 0.2  # 0 = no smoothing (raw), 1 = frozen at previous command
        v = smoothing * self.prev_v + (1.0 - smoothing) * v
        omega = smoothing * self.prev_omega + (1.0 - smoothing) * omega
        self.prev_v = v
        self.prev_omega = omega
        self.msg_vel.linear.x = v
        self.msg_vel.angular.z = omega

#-------------------------------Motion Controller's---------------------------------#
    def arm_controller(self, x, z):
        self.msg_target_arm.x = x
        self.msg_target_arm.y = 0.0
        self.msg_target_arm.z = z
        self.pub_target_arm.publish(
            self.msg_target_arm
        ) 

    def go_to_checkpoint(self):
        """Drives to a staging point offset from the stick's position --
        -0.10m along the field's X axis, +1m along Y -- before starting the
        precise final approach in motion_controller(). Keeps the actual
        pickup approach clear of whatever obstacles/path is between the
        robot's start position and the station."""
        if self.stick_x is None or self.stick_y is None:
            return
        px = self.robot_x + self.l * math.cos(self.robot_theta)
        py = self.robot_y + self.l * math.sin(self.robot_theta)
        checkpoint_x = self.stick_x - 0.10
        checkpoint_y = self.stick_y + 1.0
        ex = checkpoint_x - px
        ey = checkpoint_y - py
        distance = math.sqrt(ex**2 + ey**2)
        if distance < 0.15:
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0
            self.state = "GET_STICK"
            self.get_logger().info("REACHED CHECKPOINT; APPROACHING STICK")
            return
        ux_nom = self.kp * ex
        uy_nom = self.kp * ey
        ux, uy = self.solve_clf_cbf_qp(ux_nom, uy_nom)
        theta = self.robot_theta
        v = ux * math.cos(theta) + uy * math.sin(theta)
        omega = (-ux * math.sin(theta) + uy * math.cos(theta)) / self.STEER_GAIN_L
        v = max(min(v, 0.5), -0.5)
        omega = max(min(omega, 1.0), -1.0)
        self.apply_velocity(v, omega)

    def motion_controller(self):
        px = self.robot_x + self.l * math.cos(self.robot_theta)
        py = self.robot_y + self.l * math.sin(self.robot_theta)

        if self.stick_x is None or self.stick_y is None:
            return

        # Robot must align its X-axis position with the stick's X position The arm extends 0.20m forward. The virtual point P is 0.30m forward.
        # We offset the goal Y by (self.l - arm_reach) so the robot stops exactly close enough for the gripper to reach the stick center.
        goal_x = self.stick_x - 0.15
        goal_y = self.stick_y + 0.35


        ex = goal_x - px
        ey = goal_y - py

        distance = math.sqrt(ex**2 + ey**2)

        # Point the robot directly at the stick's center
        desired_theta = math.atan2(self.stick_y - self.robot_y, self.stick_x - self.robot_x)
        theta_error = desired_theta - self.robot_theta

        while theta_error > math.pi:
            theta_error -= 2.0 * math.pi
        while theta_error < -math.pi:
            theta_error += 2.0 * math.pi
            
        # Orientation Condition: Tightened the distance tolerance to 0.10m for higher precision 
        if distance < 0.10:
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0
            self.state = "GRAB_STICK"
            self.get_logger().info(f"Reached stick pick-up position. STATE = {self.state}")
            return

        # ux = self.kp * ex
        # uy = self.kp * ey

        ux_nom = self.kp * ex
        uy_nom = self.kp * ey

        ux, uy = (self.solve_clf_cbf_qp(ux_nom, uy_nom)) # Calculate control inputs with CBF for collision avoidance

        theta = self.robot_theta
        
        v = (ux * math.cos(theta) + uy * math.sin(theta))
        omega = (-ux * math.sin(theta) + uy * math.cos(theta)) / self.STEER_GAIN_L

        omega += 0.4 * theta_error #0.8

        v = max(min(v, 0.5), -0.5)         # Bound linear velocity to [-0.5, 0.5] m/s
        omega = max(min(omega, 1.0), -1.0) # Bound angular velocity to [-1.5, 1.5] rad/s

        # self.msg_vel.linear.x = v
        # self.msg_vel.angular.z = omega
        self.apply_velocity(v, omega)

    def grab_the_stick(self):
        now = self.get_clock().now()

        # Arm forward
        if self.action_step == 0:
            self.arm_controller(x=0.27, z=0.05)
            self.action_start_time = now
            self.action_step = 1
            self.get_logger().info("ARM FORWARD")
            return

        # Wait 2 seconds
        if self.action_step == 1:
            elapsed = (now - self.action_start_time).nanoseconds / 1e9

            if elapsed > 2.0:
                goal = GripperControl.Goal()
                goal.target_state = 2
                goal.power = 0.5
                self.gripper_action_client.send_goal_async(goal)
                self.action_start_time = now
                self.action_step = 2
                self.get_logger().info("GRIPPER CLOSE")
            return

        # Wait for gripper
        if self.action_step == 2:
            elapsed = (now - self.action_start_time).nanoseconds / 1e9

            if elapsed > 2.5:
                self.arm_controller(x=0.20, z=0.15)
                self.action_start_time = now
                self.action_step = 3
                self.get_logger().info("ARM UP")
            return

        # Wait for lift
        if self.action_step == 3:
            elapsed = (now - self.action_start_time).nanoseconds / 1e9
            if elapsed > 2.0:
                self.action_start_time = now
                self.action_step = 4
                self.get_logger().info("BACKUP")
            return

        # Backup
        if self.action_step == 4:
            elapsed = (now - self.action_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.msg_vel.linear.x = -0.10
                self.msg_vel.angular.z = 0.0
            else:
                self.msg_vel.linear.x = 0.0
                self.arm_controller(x=0.15, z=0.05)
                self.l = 0.40
                self.state = "GET_PUCK"
                self.action_step = 0
                self.get_logger().info("STICK PICKUP COMPLETE")

    def get_puck(self):
        if self.puck_x is None:
            return

        px = (self.robot_x + self.l * math.cos(self.robot_theta))
        py = (self.robot_y + self.l * math.sin(self.robot_theta))

        ex = self.puck_x - px
        ey = self.puck_y - py

        distance = math.sqrt(ex**2 + ey**2) 

        goal_dx = (self.goal_x - self.puck_x)
        goal_dy = (self.goal_y - self.puck_y)

        goal_mag = math.sqrt(goal_dx**2 + goal_dy**2)

        goal_dx /= goal_mag
        goal_dy /= goal_mag

        behind_x = (self.puck_x - 0.25 * goal_dx)
        behind_y = (self.puck_y - 0.25 * goal_dy)

        behind_error = math.sqrt((px - behind_x)**2 + (py - behind_y)**2)

        if behind_error < 0.10:
            self.state = "PUSH_TO_GOAL"
            self.get_logger().info("BEHIND PUCK")
            return

        # ux = self.kp * ex
        # uy = self.kp * ey

        ux_nom = self.kp * ex
        uy_nom = self.kp * ey

        ux, uy = (self.solve_clf_cbf_qp(ux_nom, uy_nom))
        theta = self.robot_theta

        v = (ux * math.cos(theta) + uy * math.sin(theta))
        omega = (-ux * math.sin(theta) + uy * math.cos(theta)) / self.STEER_GAIN_L

        v = max(min(v, 0.5), -0.5)
        omega = max(min(omega, 1.0), -1.0)

        self.apply_velocity(v, omega)
        
    def push_puck_to_goal(self): # Doesnot Work

        if (
            self.goal_x is None or
            self.goal_y is None or
            self.puck_x is None or
            self.puck_y is None
        ):
            return

        # Direction from puck to goal
        dx = self.goal_x - self.puck_x
        dy = self.goal_y - self.puck_y

        magnitude = math.sqrt(dx**2 + dy**2)

        if magnitude < 0.01:
            return

        dx /= magnitude
        dy /= magnitude

        # Target point BEHIND puck
        behind_distance = 0.25
        target_x = (self.puck_x - behind_distance * dx)
        target_y = (self.puck_y - behind_distance * dy)

        # Stick tip position
        px = (self.robot_x + self.l * math.cos(self.robot_theta))
        py = (self.robot_y + self.l * math.sin(self.robot_theta))

        ex = target_x - px
        ey = target_y - py

        distance = math.sqrt(ex**2 + ey**2)

        # Goal reached
        goal_distance = math.sqrt((self.goal_x - self.puck_x)**2 + (self.goal_y - self.puck_y)**2)

        if goal_distance < 0.25:
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0
            self.state = "DONE"
            self.get_logger().info("GOAL SCORED")
            return

        # Approximate Linearization
        # ux = self.kp * ex
        # uy = self.kp * ey

        ux_nom = self.kp * ex
        uy_nom = self.kp * ey

        ux, uy = (self.solve_clf_cbf_qp(ux_nom, uy_nom))
        theta = self.robot_theta

        v = (ux * math.cos(theta) + uy * math.sin(theta))

        omega = (-ux * math.sin(theta) + uy * math.cos(theta)) / self.STEER_GAIN_L

        v = max(min(v, 0.30), -0.30)
        omega = max(min(omega, 1.0), -1.0)

        # self.get_logger().info(
        #     f"Puck=({self.puck_x:.2f},{self.puck_y:.2f}) "
        #     f"Target=({target_x:.2f},{target_y:.2f}) "
        #     f"GoalDist={goal_distance:.2f}"
        # )
        
        self.apply_velocity(v, omega)

#------------------------------Goal Callback------------------------------------=---#
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected!')
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.get_logger().info('Gripper movement complete. State:')
#---------------------------------Main()-----------------------------------------=--#
    def timer_callback(self):
        # barriers = self.test_clf_cbf()
        # if barriers is None:
        #     return
        # self.debug_counter += 1
        # if self.debug_counter % 20 == 0:
        #     for rid, h, h_dot, cbf in barriers:
        #         self.get_logger().info(
        #             f"Robot {rid} "
        #             f"CBF={cbf:.3f}"
        #         )

        if self.state == "GET_CHECKPOINT":
            self.go_to_checkpoint()

        elif self.state == "GET_STICK":
            self.motion_controller()

        elif self.state == "GRAB_STICK":
            self.grab_the_stick()

        elif self.state == "GET_PUCK":
            self.get_puck()

        elif self.state == "PUSH_TO_GOAL": # Doesn't Work
            self.push_puck_to_goal()

        else:
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0

        self.pub_cmd_vel.publish(self.msg_vel)    
#-----------------------------------------------------------------------------------#

def main(args=None):
    rclpy.init(args=args)
    node = AlgorithmNode(robot_id=ROBOT_ID)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt detected. Shutting down...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
#Works[]