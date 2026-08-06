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

ROBOT_ID = 5

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
        self.CONTROL_LOOP_FREQUENCIES = 20  # Hz
        
        # State Tracking
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        #Stick state
        self.stick_x = None
        self.stick_y = None
        self.stick_theta = None
        
        # Pickup target orientation
        self.goal_theta = 0.0

        # Approximate linearization parameter
        self.kp = 0.3
        self.l = 0.20 # arm pickup distance in meters, OR 18 cm
        
        self.state = "NAVIGATE"

        self.gripper_busy = False
        self.current_gripper_state = 1  # 1 = Open, 2 = Closed
        self.arm_initialized = False
        self.state_start_time = None

        self.backup_counter = 0

        # Subscribers
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
            '/vrpn_mocap/hockey_sticks_4/pose',
            self.stick_callback,
            best_effort_qos
        )
        
        # Publishers
        self.msg_vel = Twist()
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 20)
        
        # NOTE: Target_arm_position expects geometry_msgs/Point
        self.msg_target_arm = Point()
        self.pub_target_arm = self.create_publisher(Point, target_arm_topic, 20)

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
        
        # Timer 
        self.timer = self.create_timer(1.0 / self.CONTROL_LOOP_FREQUENCIES, self.timer_callback)

    def sub_pose_callback(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.robot_theta = 2.0 * math.atan2(qz, qw)
        self.get_logger().info(
            f'x={self.robot_x:.2f}, '
            f'y={self.robot_y:.2f}, '
            f'theta={self.robot_theta:.2f}'
        )

    def stick_callback(self, msg):

        self.stick_x = msg.pose.position.x
        self.stick_y = msg.pose.position.y

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        self.stick_theta = 2.0 * math.atan2(qz, qw)

        self.get_logger().info(
            f'Stick: '
            f'x={self.stick_x:.2f}, '
            f'y={self.stick_y:.2f}, '
            f'theta={math.degrees(self.stick_theta):.1f}'
        )

    # def arm_controller(self): #Comment Not used
    #     return  # Currently disabled. Uncomment to enable arm oscillation.
        
    #     self.msg_target_arm.x = arm_value_x
    #     self.msg_target_arm.y = 0.0  # Y is ignored by the arm, but required by Point
    #     self.msg_target_arm.z = arm_value_z

    def arm_controller(self, x, z):

        self.msg_target_arm.x = x
        self.msg_target_arm.y = 0.0
        self.msg_target_arm.z = z

        self.pub_target_arm.publish(
            self.msg_target_arm
        )

    def transition_to(self, new_state):

        self.state = new_state

        self.state_start_time = (
            self.get_clock().now()
        )

    def initialize_arm_and_gripper(self):
        self.arm_controller(
            x=0.10,
            z=0.05
        )

        goal = GripperControl.Goal()
        goal.power = 0.5
        goal.target_state = 1

        self.gripper_action_client.send_goal_async(goal)

        self.arm_initialized = True

        self.get_logger().info(
            "ARM RETRACTED, GRIPPER OPEN"
        )    

    def motion_controller(self):
        if not self.arm_initialized: #Call It During Navigation
            self.initialize_arm_and_gripper()

        px = self.robot_x + self.l * math.cos(self.robot_theta)
        py = self.robot_y + self.l * math.sin(self.robot_theta)

        if self.stick_x is None:
            return

        pickup_distance = 0.20 #Where robot should stop from the stick to pick it up. 20 cm away from the stick.

        self.goal_theta = (
            self.stick_theta +
            math.pi / 2.0
        )

        goal_x = (
            self.stick_x
            - pickup_distance * math.cos(self.goal_theta)
        )

        goal_y = (
            self.stick_y
            - pickup_distance * math.sin(self.goal_theta)
        )

        ex = goal_x - px
        ey = goal_y - py

        distance = math.sqrt(ex**2 + ey**2)

        self.get_logger().info(
            f'Robot=({self.robot_x:.2f},{self.robot_y:.2f}) '
            f'Stick=({self.stick_x:.2f},{self.stick_y:.2f}) '
            f'Goal=({goal_x:.2f},{goal_y:.2f})'
        )

        if distance < 0.10:
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0
            #self.state = "MOVE_ARM"
            self.transition_to("MOVE_ARM")
            self.get_logger().info(
                'READY FOR PICKUP'
            )
            return

        ux = self.kp * ex
        uy = self.kp * ey
        theta = self.robot_theta
        
        v = (
            ux * math.cos(theta) +
            uy * math.sin(theta)
        )
        omega = (
            -ux * math.sin(theta) +
            uy * math.cos(theta)
        ) / self.l

        v = max(min(v, 0.5), -0.5)
        omega = max(min(omega, 1.5), -1.5)

        self.msg_vel.linear.x = v
        self.msg_vel.angular.z = omega

    def open_gripper(self):

        goal = GripperControl.Goal()

        goal.power = 0.5
        goal.target_state = 1

        send_goal_future = (
            self.gripper_action_client.send_goal_async(goal)
        )

        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

        self.gripper_busy = True
        #self.state = "MOVE_ARM"
        self.transition_to("ALIGN")
        
        self.get_logger().info(
            "GRIPPER OPEN"
        )

    def move_arm_to_stick(self):

        self.arm_controller(
            x=0.18,
            z=0.05
        )

        #self.state = "CLOSE_GRIPPER"
        if self.state_start_time is None:
            self.state_start_time = self.get_clock().now()

        elapsed = (
            self.get_clock().now() -
            self.state_start_time
        ).nanoseconds / 1e9

        if elapsed > 2.0:
            self.transition_to("CLOSE_GRIPPER")

        self.get_logger().info(
            "ARM FORWARD"
        )

    def align_with_stick(self):

        theta_error = (
            self.goal_theta -
            self.robot_theta
        )

        while theta_error > math.pi:
            theta_error -= 2*math.pi

        while theta_error < -math.pi:
            theta_error += 2*math.pi

        if abs(theta_error) < 0.10:

            self.msg_vel.angular.z = 0.0

            self.transition_to("MOVE_ARM")

            return

        self.msg_vel.linear.x = 0.0
        self.msg_vel.angular.z = (
            0.8 * theta_error
        )

    def close_gripper(self):

        goal = GripperControl.Goal()

        goal.power = 0.5
        goal.target_state = 2

        self.gripper_action_client.send_goal_async(
            goal
        )

        #self.state = "LIFT_ARM"
        if elapsed > 2.5:
            self.transition_to("LIFT_ARM")

        self.get_logger().info(
            "GRIPPER CLOSED"
        )

    def lift_arm(self):

        self.arm_controller(
            x=0.20,
            z=0.15
        )

        if elapsed > 2.0:
            self.transition_to("BACKUP")

        self.get_logger().info(
            "STICK PICKED"
        )

    def backup_robot(self):

        if self.backup_counter < 40:

            self.msg_vel.linear.x = -0.10
            self.msg_vel.angular.z = 0.0

            self.backup_counter += 1

        else:

            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0

            self.state = "DONE"

            self.get_logger().info(
                "STICK PICKUP COMPLETE"
            )
                    
    # def gripper_controller(self): #Comment Not used
    #     return  # Currently disabled. Uncomment to enable gripper oscillation.
    #     if self.gripper_busy:
    #         return

    #     raw_gripper_state = oscillate(min_val=1.0, max_val=2.0, step=self.step)
    #     target_state = int(round(raw_gripper_state))

    #     if target_state == self.current_gripper_state:
    #         return

    #     self.gripper_busy = True
    #     self.current_gripper_state = target_state

    #     goal = GripperControl.Goal()
    #     goal.power = 0.5 
    #     goal.target_state = target_state
        
    #     send_goal_future = self.gripper_action_client.send_goal_async(goal)
    #     send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected!')
            self.gripper_busy = False 
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.gripper_busy = False
        state_str = "Closed" if self.current_gripper_state == 2 else "Open"
        self.get_logger().info(f'Gripper movement complete. State: {state_str}')

    def timer_callback(self):
        # self.step += 0.05  
        #self.motion_controller()

        if self.state == "NAVIGATE":
            self.motion_controller()

        elif self.state == "MOVE_ARM":
            self.move_arm_to_stick()

        elif self.state == "ALIGN":
            self.align_with_stick()

        elif self.state == "CLOSE_GRIPPER":
            self.close_gripper()

        elif self.state == "LIFT_ARM":
            self.lift_arm()

        elif self.state == "BACKUP":
            self.backup_robot()

        elif self.state == "DONE":
            self.msg_vel.linear.x = 0.0
            self.msg_vel.angular.z = 0.0

        # self.arm_controller()
        # self.gripper_controller()
        self.pub_cmd_vel.publish(self.msg_vel)
        # self.pub_target_arm.publish(self.msg_target_arm)


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

#Works