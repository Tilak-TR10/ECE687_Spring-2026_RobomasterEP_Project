import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped, AccelStamped
from std_msgs.msg import ColorRGBA
from math import cos, sin, pi, sqrt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

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

        # Pubs and Subs for robots
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
        # HOCKEY OBJECTS (puck, stick, goal)
        # ------------------------------------------------------------------
        self.hockey_objects = {}  # will hold state, publishers, patches, etc.

        # Helper to create publishers for a hockey object
        def make_hockey_publishers(base_name):
            # base_name: e.g. 'hockey_puck_green' -> topics:
            #   /vrpn_mocap/hockey_puck_green/pose
            #   /vrpn_mocap/hockey_puck_green/twist
            #   /vrpn_mocap/hockey_puck_green/accel
            #   /vrpn_mocap/hockey_puck_green_2/pose
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

        # Goal (static) – placed near the right side
        goal_state = np.array([1.8, 0.0, 0.0])   # x, y, theta (theta unused)
        self.hockey_objects['goal_1'] = {
            'state': goal_state.copy(),
            'prev_state': goal_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': False,
            'publishers': make_hockey_publishers('hockey_goal_1'),
            'patches': [],   # filled later
        }

        # Puck (movable, 5 cm diameter)
        puck_state = np.array([0.0, 0.0, 0.0])
        self.hockey_objects['puck_green'] = {
            'state': puck_state.copy(),
            'prev_state': puck_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': True,
            'publishers': make_hockey_publishers('hockey_puck_green'),
            'patches': [],
        }

        # Stick (movable, with orientation)
        stick_state = np.array([-0.5, 0.5, 0.0])  # x, y, theta
        self.hockey_objects['sticks_1'] = {
            'state': stick_state.copy(),
            'prev_state': stick_state.copy(),
            'prev_time': self.get_clock().now(),
            'movable': True,
            'publishers': make_hockey_publishers('hockey_sticks_1'),
            'patches': [],
        }

        # Additional storage for velocity/accel computation
        for obj in self.hockey_objects.values():
            obj['velocity'] = np.zeros(3)        # vx, vy, vtheta
            obj['acceleration'] = np.zeros(3)
            obj['prev_velocity'] = np.zeros(3)

        # ------------------------------------------------------------------
        # Timer and final init
        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(f"Simulator started for robots: {self.ROBOT_IDS}")

        # Plots
        self.figure = []
        self.axes = []
        self.patches_robots = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_grippers = {rid: [] for rid in self.ROBOT_IDS}
        self.text_ids = {rid: [] for rid in self.ROBOT_IDS}
        self.__init_plot()
        self.__update_plot()

        # Mouse interaction attributes
        self.dragging = False
        self.dragged_object = None  # key in hockey_objects
        self.drag_offset = (0, 0)   # offset in data coords from object center

    # ----------------------------------------------------------------------
    # Plotting and mouse interaction
    # ----------------------------------------------------------------------
    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3],
                                  edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)

        # Robot patches
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

        # ------------------- Hockey objects patches ------------------------
        # Goal: a small rectangle (goal post)
        goal_data = self.hockey_objects['goal_1']
        goal_rect = patches.Rectangle((goal_data['state'][0] - 0.05, goal_data['state'][1] - 0.1),
                                      0.1, 0.2, edgecolor='blue', facecolor='lightblue', linewidth=2)
        self.axes.add_patch(goal_rect)
        goal_data['patches'].append(goal_rect)
        # Also a second pose marker (just a small dot)
        goal_dot = patches.Circle((goal_data['state'][0], goal_data['state'][1]), 0.02,
                                  color='blue', alpha=0.5)
        self.axes.add_patch(goal_dot)
        goal_data['patches'].append(goal_dot)

        # Puck: green circle (diameter 0.05)
        puck_data = self.hockey_objects['puck_green']
        puck_circle = patches.Circle((puck_data['state'][0], puck_data['state'][1]),
                                     0.025, facecolor='green', edgecolor='darkgreen', linewidth=1)
        self.axes.add_patch(puck_circle)
        puck_data['patches'].append(puck_circle)
        # A small dot for pose_2 (just another circle slightly offset or same)
        puck_dot2 = patches.Circle((puck_data['state'][0], puck_data['state'][1]),
                                   0.015, facecolor='lime', alpha=0.6)
        self.axes.add_patch(puck_dot2)
        puck_data['patches'].append(puck_dot2)

        # Stick: a polygon resembling a hockey stick
        stick_data = self.hockey_objects['sticks_1']
        # Define stick shape relative to (0,0) with orientation along +x.
        # We'll make a shaft (thin rectangle) and a blade (angled polygon).
        shaft_length = 0.15
        shaft_width = 0.025
        blade_length = 0.06
        blade_width = 0.04
        # Points in local frame (x forward, y left)
        local_pts = np.array([
            [0, -shaft_width/2],
            [shaft_length, -shaft_width/2],
            [shaft_length, shaft_width/2],
            [0, shaft_width/2],
            # blade: start from shaft end and angle down (blade is on the right)
            [shaft_length, shaft_width/2],
            [shaft_length + blade_length, shaft_width/2 + blade_width/2],
            [shaft_length + blade_length, shaft_width/2 - blade_width/2],
            [shaft_length, shaft_width/2 - blade_width/2],
        ])
        # Rotate and translate
        theta = stick_data['state'][2]
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        t = stick_data['state'][:2]
        world_pts = t + (local_pts @ R.T)
        stick_poly = patches.Polygon(world_pts, facecolor='brown', edgecolor='black', linewidth=1)
        self.axes.add_patch(stick_poly)
        stick_data['patches'].append(stick_poly)
        # A small dot for pose_2 (center of stick)
        stick_dot2 = patches.Circle((stick_data['state'][0], stick_data['state'][1]),
                                    0.015, facecolor='orange', alpha=0.6)
        self.axes.add_patch(stick_dot2)
        stick_data['patches'].append(stick_dot2)

        # Set axes limits
        margin = max(self.ROBOT_SIZE + [0.2])
        self.axes.set_xlim(self.ENV[0] - margin, self.ENV[0] + self.ENV[2] + margin)
        self.axes.set_ylim(self.ENV[1] - margin, self.ENV[1] + self.ENV[3] + margin)
        self.axes.grid()
        self.axes.axis('equal')

        # Connect mouse events
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)

        plt.ion()
        plt.show()

    def __update_plot(self):
        # Update robot patches
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

        # Update hockey patches
        for obj_key, obj_data in self.hockey_objects.items():
            state = obj_data['state']
            patches_list = obj_data['patches']
            if obj_key == 'goal_1':
                # Rectangle and dot
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
                # Recompute polygon vertices from local shape
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

    # ----------------------------------------------------------------------
    # Mouse interaction handlers
    # ----------------------------------------------------------------------
    def on_press(self, event):
        if event.inaxes != self.axes:
            return
        # Check if click is on any movable hockey object
        for obj_key, obj_data in self.hockey_objects.items():
            if not obj_data['movable']:
                continue
            # Check each patch of this object
            for patch in obj_data['patches']:
                if patch.contains(event)[0]:
                    self.dragging = True
                    self.dragged_object = obj_key
                    # Compute offset from object center (state) to click point
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
        # Clamp to environment boundaries (with some margin)
        margin = 0.05
        new_x = max(self.ENV[0] + margin, min(self.ENV[0] + self.ENV[2] - margin, new_x))
        new_y = max(self.ENV[1] + margin, min(self.ENV[1] + self.ENV[3] - margin, new_y))
        obj_data['state'][0] = new_x
        obj_data['state'][1] = new_y
        # Orientation stays unchanged during drag (could be updated if we wanted)
        self.__update_plot()

    def on_release(self, event):
        if self.dragging:
            self.dragging = False
            self.dragged_object = None
            self.drag_offset = (0, 0)

    # ----------------------------------------------------------------------
    # Existing callbacks and helpers
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Main update loop (publishes robot and hockey data)
    # ----------------------------------------------------------------------
    def update_and_publish(self):
        current_time = self.get_clock().now()

        # Update robots (unchanged)
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

        # ------------------------------------------------------------------
        # Update hockey objects and publish their topics
        # ------------------------------------------------------------------
        dt = self.DT  # use the same fixed timestep
        for obj_key, obj_data in self.hockey_objects.items():
            state = obj_data['state']
            prev_state = obj_data['prev_state']
            # Compute velocity (finite difference)
            if dt > 0:
                vel = (state - prev_state) / dt
            else:
                vel = np.zeros(3)
            # Compute acceleration (change in velocity)
            prev_vel = obj_data['prev_velocity']
            if dt > 0:
                acc = (vel - prev_vel) / dt
            else:
                acc = np.zeros(3)

            # Store for next iteration
            obj_data['prev_state'] = state.copy()
            obj_data['prev_velocity'] = vel.copy()
            obj_data['velocity'] = vel
            obj_data['acceleration'] = acc

            # Publish pose
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
            obj_data['publishers']['pose_2'].publish(pose_msg)  # same pose for _2 topic

            # Publish twist
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

            # Publish acceleration
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

        # Update plot
        self.__update_plot()

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