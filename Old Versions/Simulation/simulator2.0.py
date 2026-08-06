import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionServer  # NEW: action server
from rclpy.callback_groups import ReentrantCallbackGroup  # NEW

from geometry_msgs.msg import Twist, PoseStamped, TwistStamped, AccelStamped, Point
from std_msgs.msg import ColorRGBA
from math import cos, sin, pi, sqrt, atan2, acos
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# NEW: import the GripperControl action definition
try:
    from robomaster_msgs.action import GripperControl
except ImportError:
    # Fallback if the package is not available – define a dummy class for the action
    class GripperControl:
        class Goal:
            pass
        class Result:
            pass
        class Feedback:
            pass

# =============================================================================
# MODIFIED CLASS: 2-Link Robotic Arm Simulator (no sliders, new limits)
# =============================================================================
class RobotArmSimulator:
    """
    A 2-DOF planar robotic arm simulator that follows /robot{robot_id}/target_arm_position.
    No manual sliders – arm moves only via ROS2 commands.
    Joint limits: θ1 ∈ [0°, 180°], θ2 ∈ [0°, 180°] (measured from the x-axis, per hardware spec).
    Target (msg.x, msg.z) is relative to J1 (the arm's own mount point), matching the
    real RoboMaster EP arm's moveto(x, y) API. Elbow-down convention: link2_angle = theta1 - theta2.
    """
    def __init__(self, node, robot_id=10):
        self.node = node
        self.robot_id = robot_id

        # Robot parameters (meters)
        self.BASE_WIDTH = 0.20          # 20 cm
        self.BASE_HEIGHT = 0.16         # 16 cm
        self.L1 = 0.15                  # 15 cm
        self.L2 = 0.15                  # 15 cm

        # Joint angles (radians) – initialised to mid‑range
        self.theta1 = np.radians(45.0)   # start at 45°
        self.theta2 = np.radians(90.0)   # start at 90°

        # Joint limits (radians) — both joints move 0°-180° from the x-axis (per hardware spec)
        self.THETA1_MIN = np.radians(0.0)
        self.THETA1_MAX = np.radians(180.0)
        self.THETA2_MIN = np.radians(0.0)
        self.THETA2_MAX = np.radians(180.0)

        # NOTE: there is no joint2 offset. The elbow-down convention
        # (link2_angle = theta1 - theta2) naturally produces the correct fold
        # direction without needing an artificial angular shift.

        # Gripper opening (0..1) – will be set by the action server
        self.gripper_opening = 0.5

        # Trail buffer
        self.trail_points = []
        self.MAX_TRAIL = 200

        # End effector position (computed)
        self.ee_pos = (0.0, 0.0)

        # ROS2 subscription to target_arm_position
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.sub_target_arm = self.node.create_subscription(
            Point,
            f'/robot{self.robot_id}/target_arm_position',
            self.target_arm_callback,
            qos
        )
        self.node.get_logger().info(f"Arm simulator subscribed to /robot{self.robot_id}/target_arm_position")

        # Build the figure and widgets (no sliders)
        self.fig = None
        self.ax = None

        # Artists that will be updated
        self.link1_line = None
        self.link2_line = None
        self.joint1_circle = None
        self.joint2_circle = None
        self.ee_circle = None
        self.ee_label = None
        self.base_rect = None
        self.base_label = None
        self.trail_line = None
        self.reach_outer = None
        self.reach_inner = None
        self.theta1_text = None
        self.theta2_text = None
        self.ee_xy_text = None
        self.gripper_lines = []

        # Set up the plot
        self.initialize_plot()
        self.update_arm()   # initial draw
        plt.ion()
        plt.show(block=False)

    def initialize_plot(self):
        """Create the figure and axes – no sliders."""
        self.fig = plt.figure("2 DOF Robotic Arm Simulator", figsize=(8, 6))
        # Use full figure for the arm (no sliders)
        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.ax.set_title("2 DOF Robotic Arm Simulator (Remote Controlled)", color='white')
        self.ax.set_xlim(-0.05, 0.55)
        self.ax.set_ylim(-0.05, 0.55)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.tick_params(colors='white', labelcolor='white')
        ticks = np.arange(0, 0.55, 0.05)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.set_xticklabels([f"{int(t*100)}" for t in ticks])
        self.ax.set_yticklabels([f"{int(t*100)}" for t in ticks])
        self.ax.set_xlabel("X (cm)", color='white')
        self.ax.set_ylabel("Y (cm)", color='white')

        self.ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.set_axisbelow(True)

        # Custom axes with arrows
        self.ax.annotate('', xy=(0.52, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        self.ax.text(0.53, -0.01, 'X', color='white', fontsize=12)
        self.ax.annotate('', xy=(0, 0.52), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        self.ax.text(-0.01, 0.53, 'Y', color='white', fontsize=12)
        self.ax.text(-0.02, -0.02, 'O', color='white', fontsize=12)

        # Ground line
        self.ax.axhline(y=0, color='white', linewidth=1.5)

        # Base rectangle
        base_x, base_y = 0.0, 0.0
        self.base_rect = patches.Rectangle(
            (base_x, base_y), self.BASE_WIDTH, self.BASE_HEIGHT,
            facecolor='darkgray', edgecolor='lightgray', linewidth=1.5
        )
        self.ax.add_patch(self.base_rect)
        self.base_label = self.ax.text(base_x + self.BASE_WIDTH/2 - 0.025,
                                       base_y + self.BASE_HEIGHT/2 - 0.01,
                                       'Robot', color='white', fontsize=10, ha='center')

        # Reachability circles
        outer_radius = self.L1 + self.L2
        inner_radius = abs(self.L1 - self.L2)
        j1_x, j1_y = self.BASE_WIDTH/2.0, self.BASE_HEIGHT
        self.reach_outer = patches.Circle(
            (j1_x, j1_y), outer_radius, fill=False,
            edgecolor='gray', linestyle='--', linewidth=1, alpha=0.5
        )
        self.reach_inner = patches.Circle(
            (j1_x, j1_y), inner_radius, fill=False,
            edgecolor='gray', linestyle='--', linewidth=1, alpha=0.5
        )
        self.ax.add_patch(self.reach_outer)
        self.ax.add_patch(self.reach_inner)

        # Trail line
        self.trail_line, = self.ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7)

        # Dynamic artists
        self.link1_line, = self.ax.plot([], [], 'w-', linewidth=3)
        self.link2_line, = self.ax.plot([], [], 'w-', linewidth=3)

        # Joint1 fixed
        self.joint1_circle = patches.Circle((j1_x, j1_y), 0.008, facecolor='white')
        self.ax.add_patch(self.joint1_circle)
        self.ax.text(j1_x - 0.015, j1_y + 0.015, 'J1', color='lime', fontsize=9)

        # Joint2 (moved)
        self.joint2_circle = patches.Circle((0, 0), 0.008, facecolor='white')
        self.ax.add_patch(self.joint2_circle)
        self.joint2_label = self.ax.text(0, 0, 'J2', color='lime', fontsize=9)

        # End effector
        self.ee_circle = patches.Circle((0, 0), 0.006, facecolor='red', edgecolor='white')
        self.ax.add_patch(self.ee_circle)
        self.ee_label = self.ax.text(0, 0, 'EE', color='lime', fontsize=9)

        # Gripper: two fingers + a connecting shaft at the wrist end (3 lines total)
        self.gripper_lines = [
            self.ax.plot([], [], 'w-', linewidth=2)[0],  # top finger
            self.ax.plot([], [], 'w-', linewidth=2)[0],  # bottom finger
            self.ax.plot([], [], 'w-', linewidth=2)[0],  # connecting shaft
        ]

        # Labels
        self.theta1_text = self.ax.text(0.02, 0.50, '', color='lime', fontsize=10)
        self.theta2_text = self.ax.text(0.02, 0.47, '', color='lime', fontsize=10)
        self.ee_xy_text = self.ax.text(0.02, 0.44, '', color='lime', fontsize=10)

        # Mouse interaction placeholder (future IK)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)

    # ----------------------------------------------------------------------
    # ROS2 subscription callback
    # ----------------------------------------------------------------------
    def target_arm_callback(self, msg: Point):
        """
        Called when a new target_arm_position is published.

        IMPORTANT: msg.x and msg.z are offsets relative to J1 (the arm's own
        mount/shoulder point), matching the real RoboMaster EP arm API
        (moveto(x, y) is relative to the arm's own reference frame, not the
        world/plot origin). Do NOT subtract J1's world position again here --
        that was the root cause of the "inverted" behavior: it double-counted
        the offset and shoved every target behind/below the shoulder.
        """
        # dx, dy are already relative to J1 -- use directly, no further subtraction
        dx = msg.x
        dy = msg.z   # map ROS Z (height) to simulator's vertical axis

        # Log reception for debugging
        self.node.get_logger().info(f"Arm command received: dx={dx:.3f}, dy={dy:.3f} (relative to J1)")

        # Clamp to a sane physical envelope relative to J1 (safety)
        max_reach = self.L1 + self.L2
        dx = max(-max_reach, min(max_reach, dx))
        dy = max(-max_reach, min(max_reach, dy))

        d = sqrt(dx*dx + dy*dy)

        # Clamp to reachable range
        min_reach = abs(self.L1 - self.L2)
        if d > max_reach:
            d = max_reach
        elif d < min_reach and d > 0:
            d = min_reach

        # Elbow angle magnitude (always 0..pi from the law of cosines)
        cos_theta2 = (d*d - self.L1*self.L1 - self.L2*self.L2) / (2*self.L1*self.L2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        theta2_solution = acos(cos_theta2)   # 0..pi

        # Angle of the vector from J1 to target (relative to J1, not world origin)
        alpha = atan2(dy, dx)
        beta = atan2(self.L2*sin(theta2_solution), self.L1 + self.L2*cos(theta2_solution))

        # Elbow-DOWN solution: link2 folds back toward the ground from link1's
        # direction (theta1 + beta), matching FK's theta1 - theta2 convention below.
        # This is what lets the arm reach forward-and-down targets like a real
        # pick-and-place arm, instead of only folding further up and away.
        theta1_solution = alpha + beta

        # Clamp to joint limits
        theta1 = max(self.THETA1_MIN, min(self.THETA1_MAX, theta1_solution))
        theta2 = max(self.THETA2_MIN, min(self.THETA2_MAX, theta2_solution))

        # Update internal state
        self.theta1 = theta1
        self.theta2 = theta2

        # Redraw arm
        self.update_arm()
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # Mouse interaction (future IK)
    # ----------------------------------------------------------------------
    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.ee_circle.contains(event)[0]:
            self.ee_circle.set_edgecolor('yellow')
            self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------------
    # Forward kinematics with joint2 offset
    # ----------------------------------------------------------------------
    def compute_forward_kinematics(self):
        j1_x = self.BASE_WIDTH / 2.0
        j1_y = self.BASE_HEIGHT
        j2_x = j1_x + self.L1 * cos(self.theta1)
        j2_y = j1_y + self.L1 * sin(self.theta1)
        # Elbow-down: link2 folds back toward the ground from link1's direction.
        # Must match the theta1 = alpha + beta convention in target_arm_callback.
        link2_angle = self.theta1 - self.theta2
        ee_x = j2_x + self.L2 * cos(link2_angle)
        ee_y = j2_y + self.L2 * sin(link2_angle)
        return (j1_x, j1_y), (j2_x, j2_y), (ee_x, ee_y)

    def update_arm(self):
        """Update all dynamic artists based on current joint angles."""
        j1, j2, ee = self.compute_forward_kinematics()
        self.ee_pos = ee

        self.link1_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        self.link2_line.set_data([j2[0], ee[0]], [j2[1], ee[1]])

        self.joint2_circle.center = j2
        self.joint2_label.set_position((j2[0] + 0.015, j2[1] + 0.015))

        self.ee_circle.center = ee
        self.ee_label.set_position((ee[0] + 0.015, ee[1] + 0.015))

        # Gripper: draw as a two-finger claw.
        #   Open:    -----        Closed:   -----
        #              |
        #            -----
        # Two parallel "finger" lines extend forward from EE (along link2's
        # direction); a short shaft connects them at the wrist end. When
        # gripper_opening -> 0, the fingers and shaft all collapse onto the
        # same line, so the whole claw reads as a single flat line.
        dx = ee[0] - j2[0]
        dy = ee[1] - j2[1]
        length = sqrt(dx*dx + dy*dy)
        if length > 1e-6:
            ux = dx / length
            uy = dy / length
            px = -uy
            py = ux

            half_gap = 0.012 * self.gripper_opening   # perpendicular half-gap between fingers
            finger_length = 0.025                     # how far the fingers extend forward from EE

            top_start = (ee[0] + px * half_gap, ee[1] + py * half_gap)
            top_end = (top_start[0] + ux * finger_length, top_start[1] + uy * finger_length)
            bot_start = (ee[0] - px * half_gap, ee[1] - py * half_gap)
            bot_end = (bot_start[0] + ux * finger_length, bot_start[1] + uy * finger_length)

            self.gripper_lines[0].set_data([top_start[0], top_end[0]], [top_start[1], top_end[1]])
            self.gripper_lines[1].set_data([bot_start[0], bot_end[0]], [bot_start[1], bot_end[1]])
            self.gripper_lines[2].set_data([top_start[0], bot_start[0]], [top_start[1], bot_start[1]])
        else:
            self.gripper_lines[0].set_data([], [])
            self.gripper_lines[1].set_data([], [])
            self.gripper_lines[2].set_data([], [])

        # Trail
        self.trail_points.append(ee)
        if len(self.trail_points) > self.MAX_TRAIL:
            self.trail_points.pop(0)
        if len(self.trail_points) >= 2:
            trail_x = [p[0] for p in self.trail_points]
            trail_y = [p[1] for p in self.trail_points]
            self.trail_line.set_data(trail_x, trail_y)
        else:
            self.trail_line.set_data([], [])

        self.theta1_text.set_text(f"θ1 = {np.degrees(self.theta1):.1f}°")
        self.theta2_text.set_text(f"θ2 = {np.degrees(self.theta2):.1f}°")
        self.ee_xy_text.set_text(f"EE: ({ee[0]*100:.1f}, {ee[1]*100:.1f}) cm")

    def update_plot(self):
        """Called from main timer to refresh the arm window."""
        self.fig.canvas.draw_idle()


# =============================================================================
# NEW CLASS: Mock Gripper Action Server
# =============================================================================
class GripperActionServer:
    """
    Simple mock action server for GripperControl that accepts goals and immediately
    returns success. Updates the arm simulator's gripper opening if the robot ID
    matches the arm simulator's robot ID.
    """
    def __init__(self, node, robot_id, arm_sim=None):
        self.node = node
        self.robot_id = robot_id
        self.arm_sim = arm_sim  # only used if robot_id == arm_sim.robot_id
        self.current_state = 1   # 1 = open, 2 = closed

        # Create the action server
        self._action_server = ActionServer(
            node,
            GripperControl,
            f'/robot{robot_id}/gripper',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )
        node.get_logger().info(f"Gripper action server started for robot {robot_id}")

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        target_state = goal.target_state  # 1 or 2
        # power is ignored in simulation

        self.node.get_logger().info(f"Gripper command: target_state={target_state} for robot {self.robot_id}")

        # Update the arm simulator's gripper opening if this is the arm robot
        if self.arm_sim is not None and self.robot_id == self.arm_sim.robot_id:
            if target_state == 1:   # open
                self.arm_sim.gripper_opening = 1.0
            elif target_state == 2: # close
                self.arm_sim.gripper_opening = 0.0
            else:
                self.arm_sim.gripper_opening = 0.5

        self.current_state = target_state

        # Send feedback (optional). The real robomaster_msgs/GripperControl action
        # definition doesn't necessarily have a 'progress' field, so only set fields
        # that actually exist on this build's message — this previously crashed with
        # "AttributeError: 'GripperControl_Feedback' object has no attribute 'progress'"
        # which aborted every gripper goal.
        feedback = GripperControl.Feedback()
        if hasattr(feedback, 'progress'):
            feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)

        # Succeed immediately
        goal_handle.succeed()

        result = GripperControl.Result()
        if hasattr(result, 'success'):
            result.success = True
        if hasattr(result, 'state'):
            result.state = target_state
        return result


# =============================================================================
# EXISTING SIMULATOR (modified to include action servers)
# =============================================================================
class MultiRoboMasterSim(Node):
    def __init__(self):
        super().__init__('multi_robomaster_sim')

        # constants
        # robots
        self.ROBOT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.N = len(self.ROBOT_IDS)
        # time
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 20 # milliseconds
        self.TIMEOUT_GET_POSES = 10 # milliseconds
        self.TIMEOUT_CHASSIS_SPEED = 500 # milliseconds
        self.DT = (self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES) / 1000.
        # robot control
        self.MAX_LINEAR_SPEED = 1.0 # meters / second
        self.MAX_ANGULAR_SPEED = 360 * np.pi / 180 # radians / second
        # dimensions
        self.ENV = [-2., -2., 4., 4.] # (x, y) can vary from (ENV[0], ENV[1]) to (ENV[0]+ENV[2], ENV[1]+ENV[3])
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        self.GRIPPER_SIZE = 0.1

        # State: [x, y, theta]
        self.states = {}
        self.leds = {}
        self.velocities = {rid: np.array([0.0, 0.0, 0.0]) for rid in self.ROBOT_IDS}
        self.last_cmd_time = {rid: self.get_clock().now() for rid in self.ROBOT_IDS}

        # Initialize robots randomly
        for i, rid in enumerate(self.ROBOT_IDS):
            x = np.random.uniform(self.ENV[0], self.ENV[0] + self.ENV[2])
            y = np.random.uniform(self.ENV[1], self.ENV[1] + self.ENV[3])
            theta = np.random.random() * 2 * np.pi

            self.states[rid] = np.array([x, y, theta])
            self.leds[rid] = np.array([0., 0., 0.])

        # Pubs and Subs
        self.pubs = {}
        self.subs_vel = {}
        self.subs_led = {}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        for rid in self.ROBOT_IDS:
            # Publisher: Mimics VRPN motion capture system
            self.pubs[rid] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/dji_robot_{rid}/pose', qos)

            # Subscriber: Listen to the controller's cmd_vel
            self.subs_vel[rid] = self.create_subscription(
                Twist, f'/robot{rid}/cmd_vel',
                lambda msg, rid=rid: self.vel_callback(msg, rid), qos)

            # Subscriber: Listen to the controller's leds
            self.subs_led[rid] = self.create_subscription(
                ColorRGBA, f'/robot{rid}/leds/color',
                lambda msg, rid=rid: self.led_callback(msg, rid), qos)

        # ------------------------------------------------------------------
        # HOCKEY OBJECTS (puck, stick, goal) - unchanged
        # ------------------------------------------------------------------
        self.hockey_objects = {}  # will hold state, publishers, patches, etc.

        def make_hockey_publishers(base_name):
            pubs = {}
            pubs['pose'] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/{base_name}/pose', qos)
            pubs['twist'] = self.create_publisher(
                TwistStamped, f'/vrpn_mocap/{base_name}/twist', qos)
            pubs['accel'] = self.create_publisher(
                AccelStamped, f'/vrpn_mocap/{base_name}/accel', qos)
            pubs['pose_2'] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/{base_name}_2/pose', qos)
            return pubs

        # Goal (static)
        goal_state = np.array([1.8, 0.0, 0.0])
        self.hockey_objects['goal_1'] = {
            'state': goal_state.copy(),
            'prev_state': goal_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': False,
            'publishers': make_hockey_publishers('hockey_goal_1'),
            'patches': [],
        }

        # Puck (movable)
        puck_state = np.array([0.0, 0.0, 0.0])
        self.hockey_objects['puck_green'] = {
            'state': puck_state.copy(),
            'prev_state': puck_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': True,
            'publishers': make_hockey_publishers('hockey_puck_green'),
            'patches': [],
        }

        # Stick (movable)
        stick_state = np.array([-0.5, 0.5, 0.0])
        self.hockey_objects['sticks_1'] = {
            'state': stick_state.copy(),
            'prev_state': stick_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': True,
            'publishers': make_hockey_publishers('hockey_sticks_1'),
            'patches': [],
        }

        for obj in self.hockey_objects.values():
            obj['velocity'] = np.zeros(3)
            obj['acceleration'] = np.zeros(3)
            obj['prev_velocity'] = np.zeros(3)

        # ------------------------------------------------------------------
        # ARM SIMULATOR - now with no sliders, new limits, and debug logging
        # ------------------------------------------------------------------
        # NOTE: This must match the ROBOT_ID being commanded by the algorithm node
        # (hockey_node.py currently uses ROBOT_ID = 7). If you add more arm-controlled
        # robots later, you'll need one RobotArmSimulator per robot_id, each subscribed
        # to its own /robot{id}/target_arm_position topic.
        self.ARM_ROBOT_ID = 7
        self.arm_sim = RobotArmSimulator(self, robot_id=self.ARM_ROBOT_ID)

        # ------------------------------------------------------------------
        # GRIPPER ACTION SERVERS for all robots
        # ------------------------------------------------------------------
        self.gripper_servers = {}
        for rid in self.ROBOT_IDS:
            # Only pass arm_sim to the server if this robot ID matches the arm_sim's robot_id
            arm_ref = self.arm_sim if rid == self.arm_sim.robot_id else None
            self.gripper_servers[rid] = GripperActionServer(self, rid, arm_ref)

        # ------------------------------------------------------------------
        # Timer and final init
        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(f"Simulator started for robots: {self.ROBOT_IDS}")

        # Plots for mobile robots (unchanged)
        self.figure = []
        self.axes = []
        self.patches_robots = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_grippers = {rid: [] for rid in self.ROBOT_IDS}
        self.text_ids = {rid: [] for rid in self.ROBOT_IDS}
        self.__init_plot()
        self.__update_plot()

        # Mouse interaction for hockey (unchanged)
        self.dragging = False
        self.dragged_object = None
        self.drag_offset = (0, 0)

    # ----------------------------------------------------------------------
    # Existing plotting and interaction methods (unchanged)
    # ----------------------------------------------------------------------
    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3],
                                  edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)

        for i, rid in enumerate(self.ROBOT_IDS):
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])],
                          [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            p_robot = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                                     [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T),
                                      facecolor='k')
            p_gripper = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T),
                                        facecolor='k')
            text_id = plt.text(self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0,
                               self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0,
                               s=str(self.ROBOT_IDS[i]), color="red")
            self.patches_robots[rid] = p_robot
            self.patches_grippers[rid] = p_gripper
            self.text_ids[rid] = text_id
            self.axes.add_patch(p_robot)
            self.axes.add_patch(p_gripper)

        # Hockey objects patches (unchanged)
        goal_data = self.hockey_objects['goal_1']
        goal_rect = patches.Rectangle((goal_data['state'][0] - 0.05, goal_data['state'][1] - 0.1),
                                      0.1, 0.2, edgecolor='blue', facecolor='lightblue', linewidth=2)
        self.axes.add_patch(goal_rect)
        goal_data['patches'].append(goal_rect)
        goal_dot = patches.Circle((goal_data['state'][0], goal_data['state'][1]), 0.02,
                                  color='blue', alpha=0.5)
        self.axes.add_patch(goal_dot)
        goal_data['patches'].append(goal_dot)

        puck_data = self.hockey_objects['puck_green']
        puck_circle = patches.Circle((puck_data['state'][0], puck_data['state'][1]),
                                     0.025, facecolor='green', edgecolor='darkgreen', linewidth=1)
        self.axes.add_patch(puck_circle)
        puck_data['patches'].append(puck_circle)
        puck_dot2 = patches.Circle((puck_data['state'][0], puck_data['state'][1]),
                                   0.015, facecolor='lime', alpha=0.6)
        self.axes.add_patch(puck_dot2)
        puck_data['patches'].append(puck_dot2)

        stick_data = self.hockey_objects['sticks_1']
        local_pts = np.array([
            [0, -0.025/2],
            [0.15, -0.025/2],
            [0.15, 0.025/2],
            [0, 0.025/2],
            [0.15, 0.025/2],
            [0.15 + 0.06, 0.025/2 + 0.04/2],
            [0.15 + 0.06, 0.025/2 - 0.04/2],
            [0.15, 0.025/2 - 0.04/2],
        ])
        theta = stick_data['state'][2]
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        t = stick_data['state'][:2]
        world_pts = t + (local_pts @ R.T)
        stick_poly = patches.Polygon(world_pts, facecolor='brown', edgecolor='black', linewidth=1)
        self.axes.add_patch(stick_poly)
        stick_data['patches'].append(stick_poly)
        stick_dot2 = patches.Circle((stick_data['state'][0], stick_data['state'][1]),
                                    0.015, facecolor='orange', alpha=0.6)
        self.axes.add_patch(stick_dot2)
        stick_data['patches'].append(stick_dot2)

        margin = max(self.ROBOT_SIZE + [0.2])
        self.axes.set_xlim(self.ENV[0] - margin, self.ENV[0] + self.ENV[2] + margin)
        self.axes.set_ylim(self.ENV[1] - margin, self.ENV[1] + self.ENV[3] + margin)
        self.axes.grid()
        self.axes.axis('equal')

        self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)

        plt.ion()
        plt.show()

    def __update_plot(self):
        for rid in self.ROBOT_IDS:
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])],
                          [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            xy_robot = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                      [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T)
            xy_gripper = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T)

            self.patches_robots[rid].xy = xy_robot
            self.patches_grippers[rid].xy = xy_gripper
            self.patches_robots[rid].set_facecolor(self.leds[rid])
            self.text_ids[rid].set_position((self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0,
                                             self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0))

        for obj_key, obj_data in self.hockey_objects.items():
            state = obj_data['state']
            patches_list = obj_data['patches']
            if obj_key == 'goal_1':
                rect = patches_list[0]
                dot = patches_list[1]
                rect.set_xy((state[0] - 0.05, state[1] - 0.1))
                dot.center = (state[0], state[1])
            elif obj_key == 'puck_green':
                circle = patches_list[0]
                dot2 = patches_list[1]
                circle.center = (state[0], state[1])
                dot2.center = (state[0], state[1])
            elif obj_key == 'sticks_1':
                poly = patches_list[0]
                dot2 = patches_list[1]
                local_pts = np.array([
                    [0, -0.025/2],
                    [0.15, -0.025/2],
                    [0.15, 0.025/2],
                    [0, 0.025/2],
                    [0.15, 0.025/2],
                    [0.15 + 0.06, 0.025/2 + 0.04/2],
                    [0.15 + 0.06, 0.025/2 - 0.04/2],
                    [0.15, 0.025/2 - 0.04/2],
                ])
                theta = state[2]
                R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                t = state[:2]
                world_pts = t + (local_pts @ R.T)
                poly.set_xy(world_pts)
                dot2.center = (state[0], state[1])

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def on_press(self, event):
        if event.inaxes != self.axes:
            return
        for obj_key, obj_data in self.hockey_objects.items():
            if not obj_data['movable']:
                continue
            for patch in obj_data['patches']:
                if patch.contains(event)[0]:
                    self.dragging = True
                    self.dragged_object = obj_key
                    center = obj_data['state'][:2]
                    click = (event.xdata, event.ydata)
                    self.drag_offset = (center[0] - click[0], center[1] - click[1])
                    return

    def on_motion(self, event):
        if not self.dragging or self.dragged_object is None:
            return
        if event.inaxes != self.axes:
            return
        obj_data = self.hockey_objects[self.dragged_object]
        new_x = event.xdata + self.drag_offset[0]
        new_y = event.ydata + self.drag_offset[1]
        margin = 0.05
        new_x = max(self.ENV[0] + margin, min(self.ENV[0] + self.ENV[2] - margin, new_x))
        new_y = max(self.ENV[1] + margin, min(self.ENV[1] + self.ENV[3] - margin, new_y))
        obj_data['state'][0] = new_x
        obj_data['state'][1] = new_y
        self.__update_plot()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.dragged_object = None
            self.drag_offset = (0, 0)

    @staticmethod
    def transform_velocity_local_to_global(robots_speeds, theta):
        robots_speeds_global = [0] * 3
        x_dot = robots_speeds[0]
        y_dot = robots_speeds[1]
        th_dot = robots_speeds[2]
        c_th = cos(theta)
        s_th = sin(theta)
        robots_speeds_global[0] = c_th * x_dot - s_th * y_dot
        robots_speeds_global[1] = s_th * x_dot + c_th * y_dot
        robots_speeds_global[2] = robots_speeds[2]
        return robots_speeds_global

    def vel_callback(self, msg, rid):
        robot_speeds = MultiRoboMasterSim.transform_velocity_local_to_global(
            [msg.linear.x, msg.linear.y, msg.angular.z], self.states[rid][2])
        self.velocities[rid] = np.array(robot_speeds)
        self.last_cmd_time[rid] = self.get_clock().now()

    def led_callback(self, msg, rid):
        self.leds[rid] = np.array([msg.r, msg.g, msg.b])

    def update_and_publish(self):
        current_time = self.get_clock().now()

        # Update robots
        for rid in self.ROBOT_IDS:
            elapsed_time_since_last_command_received = (current_time - self.last_cmd_time[rid]).nanoseconds / 1e9
            if elapsed_time_since_last_command_received > self.TIMEOUT_CHASSIS_SPEED / 1e3:
                v_cmd = np.array([0.0, 0.0, 0.0])
            else:
                v_cmd = self.velocities[rid]

            self.states[rid][0] += v_cmd[0] * self.DT
            self.states[rid][1] += v_cmd[1] * self.DT
            self.states[rid][2] += v_cmd[2] * self.DT

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.pose.position.x = self.states[rid][0]
            msg.pose.position.y = self.states[rid][1]
            msg.pose.position.z = 0.0
            half_yaw = self.states[rid][2] * 0.5
            msg.pose.orientation.z = sin(half_yaw)
            msg.pose.orientation.w = cos(half_yaw)
            self.pubs[rid].publish(msg)

        # Update hockey objects
        dt = self.DT
        for obj_key, obj_data in self.hockey_objects.items():
            state = obj_data['state']
            prev_state = obj_data['prev_state']
            if dt > 0:
                vel = (state - prev_state) / dt
            else:
                vel = np.zeros(3)
            prev_vel = obj_data['prev_velocity']
            if dt > 0:
                acc = (vel - prev_vel) / dt
            else:
                acc = np.zeros(3)

            obj_data['prev_state'] = state.copy()
            obj_data['prev_velocity'] = vel.copy()
            obj_data['velocity'] = vel
            obj_data['acceleration'] = acc

            pose_msg = PoseStamped()
            pose_msg.header.stamp = current_time.to_msg()
            pose_msg.header.frame_id = 'world'
            pose_msg.pose.position.x = state[0]
            pose_msg.pose.position.y = state[1]
            pose_msg.pose.position.z = 0.0
            half_yaw = state[2] * 0.5
            pose_msg.pose.orientation.z = sin(half_yaw)
            pose_msg.pose.orientation.w = cos(half_yaw)
            obj_data['publishers']['pose'].publish(pose_msg)
            obj_data['publishers']['pose_2'].publish(pose_msg)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.header.frame_id = 'world'
            twist_msg.twist.linear.x = vel[0]
            twist_msg.twist.linear.y = vel[1]
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = vel[2]
            obj_data['publishers']['twist'].publish(twist_msg)

            accel_msg = AccelStamped()
            accel_msg.header.stamp = current_time.to_msg()
            accel_msg.header.frame_id = 'world'
            accel_msg.accel.linear.x = acc[0]
            accel_msg.accel.linear.y = acc[1]
            accel_msg.accel.linear.z = 0.0
            accel_msg.accel.angular.x = 0.0
            accel_msg.accel.angular.y = 0.0
            accel_msg.accel.angular.z = acc[2]
            obj_data['publishers']['accel'].publish(accel_msg)

        # Update mobile robot plot
        self.__update_plot()

        # Update the arm simulator window
        self.arm_sim.update_plot()


def main(args=None):
    rclpy.init(args=args)
    node = MultiRoboMasterSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()