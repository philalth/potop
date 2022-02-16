"""
Rendering functionality for the environment.
"""

import pyglet
from envs import custom_rendering

from envs.utils import get_min_max
from envs.enums import ParkingStatus
from config.settings import PARTIAL_RENDERING, AGENTS_VIEW, SCREEN_HEIGHT, SCREEN_WIDTH, \
    SAVE_INTERVAL, VIDEO_FOLDER


# pylint: disable=R0903
class DrawText:
    """Describes the status of the parking space."""

    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self) -> None:
        """Renders the text to the environment"""
        self.label.draw()


# pylint: disable=R0902
class EnvironmentVisualization:
    """
    The environments visualization.
    """

    def __init__(self):
        self.spot_visualization_list = []
        self.agent_transition_list = []
        self.viewer = None
        self.agent_transitions = []
        self.partial_rendering = PARTIAL_RENDERING
        self.agents_view = AGENTS_VIEW
        self.date = None
        self.day_count = 0

    def render(self, graph, spots, agent_positions, current_time) -> None:
        """Returns the visual representation of the environment."""
        num_agents = len(agent_positions)
        min_x, max_x, min_y, max_y = get_min_max(graph)

        scale_x = SCREEN_WIDTH / (max_x - min_x)
        scale_y = SCREEN_HEIGHT / (max_y - min_y)
        # take smallest scale value to evenly scale along x and y axis
        scale = min(scale_x, scale_y)
        if self.viewer is None:
            self.viewer = custom_rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            self._draw_map(graph, min_x, min_y, scale)
            self._draw_agents(num_agents)

        self._colorize_parking_spots(spots)

        # set the location of the agents
        for index, agent_trans in enumerate(self.agent_transition_list):
            agent_trans.set_translation(
                (agent_positions[index][0] - min_x) * scale,
                (agent_positions[index][1] - min_y) * scale)

        self._draw_current_time(current_time)
        if self.date is None:
            self.date = current_time.date()
        if self.date != current_time.date():
            self.date = current_time.date()
            self.day_count += 1

        return self.viewer.render(return_rgb_array='human' == 'rgb_array',
                                  save_images=((self.day_count % SAVE_INTERVAL) == 0),
                                  path=(VIDEO_FOLDER + "/" + str(self.date)))

    # pylint: disable=R0914
    # to be removed in later fixes
    def _draw_map(self, graph, min_x, min_y, scale):
        # give an ID to every parking spot to recolor them later
        spot_id = 0
        for source_node, destination_node, data in graph.edges(data=True):

            destination_node_x, destination_node_y, source_node_x, source_node_y = \
                self._draw_street(destination_node, min_x, min_y, scale, source_node)

            if 'spots' not in data:
                continue

            spot_len = len(data.get('spots'))
            spot_position_on_edge = 1
            # position the visual parking spots along its edge
            for _ in data.get('spots'):
                spot_vis = custom_rendering.make_circle(2)  # radius of the circle
                spot_vis.set_color(0, 1, 0)
                spot_trans = custom_rendering.Transform()
                spot_vis.add_attr(spot_trans)
                self.spot_visualization_list.append((spot_id, spot_vis))
                self.viewer.add_geom(spot_vis)
                x_value = source_node_x - ((source_node_x - destination_node_x) / (
                        1 + spot_len)) * spot_position_on_edge
                y_value = source_node_y - ((source_node_y - destination_node_y) / (
                        1 + spot_len)) * spot_position_on_edge
                spot_trans.set_translation((x_value - min_x) * scale,
                                           ((y_value - min_y) * scale))
                spot_position_on_edge += 1
                spot_id += 1

    # pylint: disable=R0913
    # to be removed in later fixes
    def _draw_street(self, destination_node, min_x, min_y, scale, source_node):
        source_node_x, source_node_y = source_node
        destination_node_x, destination_node_y = destination_node
        street = custom_rendering.Line(
            ((source_node_x - min_x) * scale,
             (source_node_y - min_y) * scale),
            ((destination_node_x - min_x) * scale,
             (destination_node_y - min_y) * scale))
        self.viewer.add_geom(street)
        return destination_node_x, destination_node_y, source_node_x, source_node_y

    def _draw_current_time(self, current_time):
        current_time = current_time.strftime("%H:%M:%S %d.%m.%Y")
        date = custom_rendering.pyglet.text.Label(current_time, font_size=20,
                                                  x=SCREEN_WIDTH - 8, y=SCREEN_HEIGHT - 8,
                                                  anchor_x='right', anchor_y='top',
                                                  color=(0, 0, 0, 255))
        self.viewer.add_onetime(DrawText(date))

    def _draw_agents(self, num_agents):
        for _ in range(num_agents):
            left, right, top, bottom = -10, 10, -10, 10
            agent = custom_rendering.FilledPolygon(
                [(left, bottom), (left, top), (right, top), (right, bottom)])
            agent.set_color(.8, .8, 0)
            agent.add_attr(custom_rendering.Transform(translation=(0, 0)))
            agent_trans = custom_rendering.Transform()
            agent.add_attr(agent_trans)
            self.agent_transition_list.append(agent_trans)
            self.viewer.add_geom(agent)

    def _colorize_parking_spots(self, spots):
        spot_id = 0
        for parking_status in spots:
            spot_visualization = self.spot_visualization_list[spot_id][1]

            if parking_status == ParkingStatus.FREE:
                spot_visualization.set_color(0, 1, 0)
            elif parking_status == ParkingStatus.OCCUPIED:
                spot_visualization.set_color(0, 0, 1)
            elif parking_status == ParkingStatus.IN_VIOLATION:
                spot_visualization.set_color(1, 0, 0)
            elif parking_status == ParkingStatus.FINED:
                spot_visualization.set_color(1, 1, 0)
            elif parking_status == ParkingStatus.UNKNOWN:
                spot_visualization.set_color(0, 0, 0)
            else:
                raise AssertionError

            spot_id += 1
