import heapq
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# 定义八个方向（包括斜向）
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] # 斜向


class Path_Planner():
    def __init__(
        self,
        map_free,
        map_resolution: float,
        step_size_m: float = 0.5,
        clearance_m: float = 0.2,
        smooth=False,
        viz=False,
        verbose: bool = True,
    ):
        self.map_resolution = float(map_resolution)
        self.step_size_m = float(step_size_m)
        self.clearance_m = float(clearance_m)
        self.step_px = max(1, int(round(self.step_size_m / self.map_resolution)))
        self.dilate_px = max(1, int(round(self.clearance_m / self.map_resolution)))
        self.smooth = smooth
        self.map_dialate = self.post_proc_map(map_free)
        self.map_size = self.map_dialate.shape
        self.directions = self._build_directions(self.step_px)

        self.viz = viz
        self.verbose = verbose

    def post_proc_map(self, map):
        kernel = np.ones((self.dilate_px, self.dilate_px), np.uint8)
        dilated_img = cv2.dilate(map, kernel, iterations=1)

        return dilated_img

    def get_astar_path(self, start, goal):

        start = tuple(start)
        goal = tuple(goal)

        distance, parents= self.get_cost(start, goal)

        # 重建路径并可视化
        if distance[goal] != float('inf'):
            path = self.reconstruct_path(parents, start, goal)
            # control_points = self.select_control_points(path)
            control_points = path

            if self.smooth:

                # 使用B样条平滑控制点路径
                smoothed_path = self.smooth_with_b_spline(control_points, path.shape[0])
                print('[Astar] Found b-spline path with: ', smoothed_path.shape)

                # 可视化结果
                if self.viz:

                    self.viz_cost_and_path(distance, start, goal, smoothed_path)
                    # plt.scatter(control_points[:,1], control_points[:,0], color='g')

                return smoothed_path
            else:
                return control_points
        else:
            if self.verbose:
                print("[Astar] !!! No path found from start to goal.")
            if self.viz:
                self.viz_cost_and_path(distance, start, goal, path=None)
            return None



    def get_cost(self, start, goal):

        u_lim, v_lim = self.map_size
        distance = np.full((u_lim, v_lim), np.inf)  # 用于存储从起点到每个节点的最短距离
        distance[start] = 0
        pq = [(0, start)]  # 优先队列（最小堆），存储 (距离, 坐标)
        parent = np.full((u_lim, v_lim, 2), -1)  # 记录父节点坐标

        while pq:
            current_dist, current_node = heapq.heappop(pq)
            goal_dist = np.sqrt((current_node[0]-goal[0])**2+(current_node[1]-goal[1])**2)
            current_dist = current_dist - goal_dist

            if current_node == goal:
                break  # 到达终点
            if goal_dist <= self.step_px and self._edge_is_free(current_node, goal):
                distance[goal] = current_dist + goal_dist
                parent[goal] = current_node
                break

            for direction in self.directions:
                neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])

                if (
                    0 <= neighbor[0] < u_lim
                    and 0 <= neighbor[1] < v_lim
                    and self.map_dialate[neighbor] == 0
                    and self._edge_is_free(current_node, neighbor)
                ):
                    # 计算从当前节点到邻居节点的距离
                    weight = np.linalg.norm(direction)
                    heuristic = np.sqrt((neighbor[0]-goal[0])**2+(neighbor[1]-goal[1])**2)
                    distance_through_current = current_dist + weight + heuristic

                    if distance_through_current < distance[neighbor]:
                        distance[neighbor] = distance_through_current
                        parent[neighbor] = current_node
                        heapq.heappush(pq, (distance_through_current, neighbor))

        return distance, parent

    @staticmethod
    def _build_directions(step_px):
        return [
            (dy * step_px, dx * step_px)
            for dy, dx in DIRECTIONS
        ]

    def _edge_is_free(self, start, goal):
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        delta = goal - start
        steps = int(max(abs(delta[0]), abs(delta[1])))
        if steps <= 1:
            return True
        ys = np.rint(np.linspace(start[0], goal[0], steps + 1)).astype(np.int64)
        xs = np.rint(np.linspace(start[1], goal[1], steps + 1)).astype(np.int64)
        return bool(np.all(self.map_dialate[ys, xs] == 0))

    def reconstruct_path(self, parent, start, goal):

        # 从终点回溯到起点，重建最短路径
        x, y = goal
        path = []
        while parent[x, y][0] != -1:
            path.append((x, y))
            x, y = parent[x, y]
        path.append(start)
        path.reverse()
        return np.array(path)

    def calculate_curvature(self, path):
        # 获取x和y坐标
        x = path[:, 0]
        y = path[:, 1]

        # 计算三角形的面积部分 (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)
        dx1 = x[2:] - x[:-2]  # x3 - x1
        dy1 = y[2:] - y[:-2]  # y3 - y1
        dx2 = x[1:-1] - x[:-2]  # x2 - x1
        dy2 = y[1:-1] - y[:-2]  # y2 - y1

        # 计算曲率的分子部分
        area = np.abs(dx1 * dy2 - dy1 * dx2)

        # 计算路径长度的3/2次方部分 (x2 - x1)^2 + (y2 - y1)^2
        length_sq = (dx2 ** 2 + dy2 ** 2) ** 1.5

        # 计算曲率，避免除零错误
        curvatures = np.divide(2 * area, length_sq, where=length_sq != 0)

        return curvatures

    def select_control_points(self, path, curvature_threshold=0.2):
        curvatures = self.calculate_curvature(path)
        control_points = []  # 始终包括起点 path[0]
        j = 0
        for i, curv in enumerate(curvatures):
            if curv > curvature_threshold:
                control_points.append(path[i + 1])  # 曲率变化较大的点作为控制点
                j = 0
            else:
                j += 1
                if j > 4:
                    control_points.append(path[i])
                    j = 0
        control_points.append(path[-1])
        return np.array(control_points)


    def smooth_with_b_spline(self, control_points, num):

        tck, u = splprep(control_points.T, s=0)  # 使用B样条拟合
        x = np.linspace(0, 1, num)
        new_points = splev(x, tck)  # 插值100个点
        return np.array(new_points).T


    def viz_cost_and_path(self, distance, start, goal, path):
        plt.figure()
        plt.imshow(self.map_dialate)
        plt.imshow(distance)
        plt.plot(start[1], start[0],'gx')
        plt.plot(goal[1], goal[0], 'rx')
        if path is not None:
            plt.plot( path[:, 1], path[:, 0], color='b')
        # plt.show()
