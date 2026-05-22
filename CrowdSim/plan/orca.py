import numpy as np
from numpy.linalg import norm
from typing import List

def det(a, b):
    """ 计算二维向量 a 和 b 的行列式，等价于它们的叉积 """
    return a[0] * b[1] - a[1] * b[0]


class ORCA():
    def __init__(self, map_free, cfg):

        self.map = map_free #load_image(image, 1/cfg['map']['resolution_world']  )
        self.map_resolution = cfg['map']['resolution_viz']
        self.name='orca'
        self.dt = cfg['env']['dt']  
        self.size = cfg['agent']['radius'] + 0.6*cfg['env']['safe_distance']
        self.v_max = cfg['agent']['max_vel']  
        self.opt = False

    def get_nearby_obstacles(self, pos, radius=10, max_obstacles=10):
        """
        获取坐标 ±radius 范围内的障碍物，并按距离排序返回最近的 max_obstacles 个
        :param map_free: 二值障碍物地图 (0=自由, 1=障碍)
        :param pos: 中心位置 (x, y) 单位：网格
        :param radius: 搜索半径（网格数）
        :param max_obstacles: 最多返回的障碍物数量
        :return: 障碍物坐标数组 [[x1,y1], [x2,y2], ...]
        """
        x, y = int(round(self.map_resolution*pos[0])), int(round(self.map_resolution*pos[1]))
        h, w = self.map.shape

        # 计算切片边界（防止越界）
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)

        # 提取目标区域并获取障碍物坐标
        region = self.map[x_min:x_max, y_min:y_max]
        obs_rows, obs_cols = np.where(region == 1)
        obs_coords = np.column_stack((obs_rows + x_min, obs_cols + y_min))

        # 如果没有障碍物，直接返回空数组
        if len(obs_coords) == 0:
            return None

        # 计算每个障碍物到中心点的距离
        distances = np.linalg.norm(obs_coords - pos, axis=1)

        # 按距离排序并选择最近的 max_obstacles 个
        nearest_indices = np.argsort(distances)[:max_obstacles]
        selected_obstacles = obs_coords[nearest_indices]/self.map_resolution

        return selected_obstacles


    def get_action(self, ego_state, nbrs_state, obstacles):

        ego_pos, _, ego_vel, ego_goal = ego_state
        nbrs_idx, reldis, relpos, relvel = nbrs_state    
        vel_des = self.v_max * (ego_goal - ego_pos) / np.linalg.norm(ego_goal - ego_pos)   

        num_map_vo = 0

        if obstacles is not None:
            relobs = obstacles - ego_pos
            num_map_vo = obstacles.shape[0]


        if len(nbrs_idx) == 0 and num_map_vo == 0:
            return vel_des, [vel_des, num_map_vo, []]
        elif len(nbrs_idx)> 0 and num_map_vo == 0:
            directions, points = self.compute_vo_nbrs(relpos, reldis, relvel, ego_vel)
        elif num_map_vo > 0 and len(nbrs_idx) == 0:
            directions, points = self.compute_static_vo(relobs, ego_vel)
        else:
            directions, points = self.compute_vo_nbrs(relpos, reldis, relvel, ego_vel)
            directions_map, points_map = self.compute_static_vo(relobs, ego_vel)
            directions = np.concatenate((directions, directions_map), axis=0)
            points = np.concatenate((points, points_map), axis=0)
        
        lp_fail, line_idx_fail, vel_sol = self.linear_program2(directions, points, vel_des, False)
        if lp_fail:
            vel_sol = self.linear_program3(directions, points, line_idx_fail, num_map_vo, vel_sol)
        # print('[orca] vel_sol: ', num_map_vo)
        return vel_sol, [vel_des, num_map_vo, obstacles]


    def compute_vo_nbrs(self, relpos: np.ndarray, reldis:np.ndarray, relvel: np.ndarray, ego_vel):
        num_vo = relpos.shape[0]
        combined_radius = 2 * self.size

        directions = np.zeros_like(relpos)
        u_all = np.zeros_like(relpos)

        for line_id in range(num_vo):
            p_ab = relpos[line_id]
            v_ba = relvel[line_id]
            dis_ab = reldis[line_id]

            if dis_ab > combined_radius:
                # No collision.
                w = v_ba - p_ab
                w_length = norm(w)

                dot_wp = np.dot(w, p_ab)

                if dot_wp < 0.0 and abs(dot_wp) > combined_radius * w_length:
                    # Project on cut-off circle.
                    # cond1: rp的反方向 cond2 ： 切线垂足范围以内
                    unit_w = w / w_length
                    directions[line_id] = np.array([unit_w[1], -unit_w[0]])
                    u_all[line_id] = (combined_radius - w_length) * unit_w
                else:
                    # 切线的单位向量，下两个方向相反以确定u的方向
                    # Project on legs.
                    leg = np.sqrt(dis_ab**2 - combined_radius**2)
                    if det(p_ab, w) > 0.0:
                        # Project on left leg.
                        directions[line_id] = np.array([
                            (p_ab[0] * leg - p_ab[1] * combined_radius),
                            (p_ab[0] * combined_radius + p_ab[1] * leg)
                            ])/ (dis_ab**2)
                    else:
                        # Project on right leg.
                        directions[line_id] = -np.array([
                            (p_ab[0] * leg + p_ab[1] * combined_radius),
                            (-p_ab[0] * combined_radius + p_ab[1] * leg)
                            ])/ (dis_ab**2)
                    
                    dot_product2 = np.dot(v_ba, directions[line_id])

                    u_all[line_id] = dot_product2 * directions[line_id] - v_ba
            else:
                # Collision. Project on cut-off circle of time step.
                inv_time_step = 1.0 / self.dt

                # Vector from cutoff center to relative velocity.
                w = v_ba - inv_time_step * p_ab

                w_length = norm(w)
                unit_w = w / w_length

                directions[line_id] = np.array([unit_w[1], -unit_w[0]])
                u_all[line_id] = (combined_radius * inv_time_step - w_length) * unit_w

            points = ego_vel + 0.5 * u_all
        
        return directions, points


    def compute_static_vo(self, static_obstacles, ego_vel):
        num_vo = static_obstacles.shape[0]
        combined_radius = self.size

        directions = np.zeros_like(static_obstacles)
        u_all = np.zeros_like(static_obstacles)

        for line_id in range(num_vo):
            p_ab = static_obstacles[line_id]
            v_ba = ego_vel
            dis_ab = np.sqrt(p_ab[0]**2+p_ab[1]**2)

            if dis_ab > combined_radius:
                # No collision.
                w = v_ba - p_ab
                w_length = norm(w)

                dot_wp = np.dot(w, p_ab)

                if dot_wp < 0.0 and abs(dot_wp) > combined_radius * w_length:
                    # Project on cut-off circle.
                    # cond1: rp的反方向 cond2 ： 切线垂足范围以内
                    unit_w = w / w_length
                    directions[line_id] = np.array([unit_w[1], -unit_w[0]])
                    u_all[line_id] = (combined_radius - w_length) * unit_w
                else:
                    # 切线的单位向量，下两个方向相反以确定u的方向
                    # Project on legs.
                    leg = np.sqrt(dis_ab**2 - combined_radius**2)
                    if det(p_ab, w) > 0.0:
                        # Project on left leg.
                        directions[line_id] = np.array([
                            (p_ab[0] * leg - p_ab[1] * combined_radius),
                            (p_ab[0] * combined_radius + p_ab[1] * leg)
                            ])/ (dis_ab**2)
                    else:
                        # Project on right leg.
                        directions[line_id] = -np.array([
                            (p_ab[0] * leg + p_ab[1] * combined_radius),
                            (-p_ab[0] * combined_radius + p_ab[1] * leg)
                            ])/ (dis_ab**2)
                    
                    dot_product2 = np.dot(v_ba, directions[line_id])

                    u_all[line_id] = dot_product2 * directions[line_id] - v_ba

            points = ego_vel + u_all

        return directions, points

    def linear_program2(self, directions, points, vel_opt, opt):

        if opt:
            result = vel_opt * self.v_max
        elif norm(vel_opt) > self.v_max:
            result = vel_opt / norm(vel_opt) * self.v_max
        else:
            result = vel_opt.copy()
        # print('[LP2] result: ', vel_opt, result)
        
        for i in range(directions.shape[0]):
            # print('[LP2] det Line: ', i, det(directions[i], points[i] - result), directions[i], points[i], result)
            if det(directions[i], points[i] - result) > 0.0:
                # Result does not satisfy constraint i. Compute new optimal result.
                tempResult = result
                success, result = self.linear_program1(directions, points, i, vel_opt, opt)
                if not success:
                    result = tempResult
                    # print('[LP1] ! line fail ', i, result)
                    return True, i, result

        return False, 0, result


    def linear_program1(self, directions, points, line_idx, vel_opt, opt):
        """
        修改后的单约束线性规划
        
        参数:
            directions: (N,2)矩阵
            points: (N,2)矩阵
            line_no: 当前处理的约束线索引
            vel_opt: (2,)当前最优速度
            opt: 是否优化方向
        """
        direction = directions[line_idx]
        point = points[line_idx]
        
        dot_product = np.dot(point, direction)
        discriminant = dot_product**2 + self.v_max**2 - np.dot(point, point)
        if discriminant < 0.0:
            # Max speed circle fully invalidates line lineNo
            return False, None

        sqrtDiscriminant = np.sqrt(discriminant)
        tLeft = -dot_product - sqrtDiscriminant
        tRight = -dot_product + sqrtDiscriminant


        for i in range(directions.shape[0]):
            if i >= line_idx:
                continue
            denominator = det(direction, directions[i])
            numerator = det(directions[i], point - points[i])

            if np.abs(denominator) <= 1e-6:  # RVO_EPSILON is very small, use small threshold
                if numerator < 0.0:
                    return False, None
                else:
                    continue
                    
            t = numerator / denominator

            if denominator >= 0.0:
                # Line i bounds line lineNo on the right
                tRight = min(tRight, t)
            else:
                # Line i bounds line lineNo on the left
                tLeft = max(tLeft, t)

            if tLeft > tRight:
                return False, None

        # Now calculate the result based on direction optimization
        if opt:
            if np.dot(vel_opt, direction) > 0.0:
                result = point + tRight * direction
            else:
                result = point + tLeft * direction
            # print('[LP1 opt]', result)
        else:
            t = np.dot(direction, vel_opt - point)

            if t < tLeft:
                result = point + tLeft * direction
            elif t > tRight:
                result = point + tRight * direction
            else:
                result = point + t * direction

        return True, result

    def linear_program3(self, directions, points, line_idx_fail, num_obs_lines, result):
        distance = 0.0

        for i in range(line_idx_fail, directions.shape[0]):

            if det(directions[i], points[i] - result) > distance:
                directions_new = []
                points_new = []

                for j in range(num_obs_lines, i):
                    determinant = det(directions[i], directions[j])
                    # print('[LP3] new optimization', determinant)
                    if np.abs(determinant) <= 1e-6:
                        # Line i and line j are parallel.
                        if np.dot(directions[i], directions[j]) > 0.0:
                            # Line i and line j point in the same direction.
                            continue
                        else:
                            # Line i and line j point in opposite direction.
                           point_new = 0.5 * (points[i] + points[j])
                    else:
                        point_new = points[i] + (det(directions[j], points[i] - points[j]) / determinant) * directions[i]
                    direction_new = (directions[j] - directions[i])/np.linalg.norm(directions[j] - directions[i])
                    directions_new.append(direction_new)
                    points_new.append(point_new)

                opt_dir = np.array([-directions[i][1], directions[i][0]])
                _, _, result = self.linear_program2(np.array(directions_new), np.array(points_new), opt_dir, True)

                distance = det(directions[i], points[i] - result)

        return result