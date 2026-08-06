import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from rosidl_runtime_py import message_to_yaml
from geometry_msgs.msg import Twist, PoseStamped, Point
from robomaster_msgs.action import GripperControl

ROBOT_ID = 7


# Just for testing
def oscillate(min_val, max_val, step):
    """Calculates a value oscillating smoothly between min_val and max_val."""
    mid = (max_val + min_val) / 2.0
    amplitude = (max_val - min_val) / 2.0
    return mid + amplitude * math.sin(step)


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
        self.CONTROL_LOOP_FREQUENCIES = 10  # Hz
        
        # State Tracking
        self.robot_pose = [None] * 3  # x, y, theta
        self.step = 0.0
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
        
        # Publishers
        self.msg_vel = Twist()
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        # NOTE: Target_arm_position expects geometry_msgs/Point
        self.msg_target_arm = Point()
        self.pub_target_arm = self.create_publisher(Point, target_arm_topic, 10)

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

    def sub_pose_callback(self, msg: PoseStamped):
        self.robot_pose = [0.0, 0.0, 0.0]  # placeholder
        
        # Uncomment these lines if you need to debug pose data. 
        # Left commented out by default to avoid flooding the terminal at high Hz.
        # self.get_logger().info('in sub pose listener')
        # self.get_logger().info(f'Received pose: \n{message_to_yaml(msg)}')

    def arm_controller(self):
        arm_value_x = oscillate(min_val=0.0, max_val=0.2, step=self.step)
        arm_value_z = oscillate(min_val=0.0, max_val=0.2, step=self.step)
        
        self.msg_target_arm.x = arm_value_x
        self.msg_target_arm.y = 0.0  # Y is ignored by the arm, but required by Point
        self.msg_target_arm.z = arm_value_z

    def motion_controller(self):
        self.msg_vel.linear.x = oscillate(min_val=-0.1, max_val=0.1, step=self.step)
        self.msg_vel.angular.z = oscillate(min_val=-1.0, max_val=1.0, step=self.step)

    def gripper_controller(self):
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
        self.step += 0.05  

        self.motion_controller()
        self.arm_controller()
        self.gripper_controller()

        self.pub_cmd_vel.publish(self.msg_vel)
        self.pub_target_arm.publish(self.msg_target_arm)


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