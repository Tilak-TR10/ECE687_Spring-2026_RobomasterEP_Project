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

ROBOT_ID = 4

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
        
        # Pickup target orientation
        self.goal_theta = 0.0

        # Approximate linearization parameter
        self.kp = 0.8
        self.l = 0.10 # in meters, OR 10 cm 
        
        self.gripper_busy = False
        self.current_gripper_state = 1  # 1 = Open, 2 = Closed

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
            '/vrpn_mocap/hockey_sticks_1/pose',
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
        # self.gripper_action_client = ActionClient(
        #     self,
        #     GripperControl,
        #     gripper_action,
        #     callback_group=self._action_group
        # )
        
        # Wait for the action server to be online before starting
        self.get_logger().info(f'Waiting for action server: {gripper_action}...')
        #self.gripper_action_client.wait_for_server()
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

    def arm_controller(self): #Comment Not used
        return  # Currently disabled. Uncomment to enable arm oscillation.
        arm_value_x = oscillate(min_val=0.0, max_val=0.2, step=self.step)
        arm_value_z = oscillate(min_val=0.0, max_val=0.2, step=self.step)
        
        self.msg_target_arm.x = arm_value_x
        self.msg_target_arm.y = 0.0  # Y is ignored by the arm, but required by Point
        self.msg_target_arm.z = arm_value_z

    def motion_controller(self):
        px = self.robot_x + self.l * math.cos(self.robot_theta)
        py = self.robot_y + self.l * math.sin(self.robot_theta)


        if self.stick_x is None:
            return

        pickup_distance = 0.20

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
            self.get_logger().info('TARGET REACHED')
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

    def gripper_controller(self): #Comment Not used
        return  # Currently disabled. Uncomment to enable gripper oscillation.
        if self.gripper_busy:
            return

        raw_gripper_state = oscillate(min_val=1.0, max_val=2.0, step=self.step)
        target_state = int(round(raw_gripper_state))

        if target_state == self.current_gripper_state:
            return

        self.gripper_busy = True
        self.current_gripper_state = target_state

        goal = GripperControl.Goal()
        goal.power = 0.5 
        goal.target_state = target_state
        
        send_goal_future = self.gripper_action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

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

        self.motion_controller()
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