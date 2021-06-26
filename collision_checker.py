#!/usr/bin/env python3
import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight

    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    def collision_check(self, paths, obstacles, obstacles_type,follow_lead_vehicle):
        """Returns a bool array on whether each path is collision free.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame.
                Format: [[x0, y0],
                         [x1, y1],
                         ...,
                         [xn, yn]]
                , where n is the number of obstacle points and units are [m, m]

        returns:
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
        """
        collision_check_array = np.zeros(len(paths), dtype=bool)
        stopped_at_pedestrian = False
        for i in range(len(paths)):
            collision_free = True
            path           = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):
                # Compute the circle locations along this point in the path.
                # These circle represent an approximate collision
                # border for the vehicle, which will be used to check
                # for any potential collisions along each path with obstacles.

                # The circle offsets are given by self._circle_offsets.
                # The circle offsets need to placed at each point along the path,
                # with the offset rotated by the yaw of the vehicle.
                # Each path is of the form [[x_values], [y_values],
                # [theta_values]], where each of x_values, y_values, and
                # theta_values are in sequential order.

                # Thus, we need to compute:
                # circle_x = point_x + circle_offset*cos(yaw)
                # circle_y = point_y circle_offset*sin(yaw)
                # for each point along the path.
                # point_x is given by path[0][j], and point _y is given by
                # path[1][j]. 
                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offset = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offset * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offset * sin(path[2][j])

                # Assumes each obstacle is approximated by a collection of
                # points of the form [x, y].
                # Here, we will iterate through the obstacle points, and check
                # if any of the obstacle points lies within any of our circles.
                # If so, then the path will collide with an obstacle and
                # the collision_free flag should be set to false for this flag
                for k in range(len(obstacles)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles[k], 
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists, 
                                                  self._circle_radii)
                    collision_free = collision_free and \
                                     not np.any(collision_dists < 0)
                    
                    if(obstacles_type[k] == 'pedestrian' and collision_free == False and i <len(paths)-5 and i > 3 ):
                        stopped_at_pedestrian = True

                    if not collision_free:
                        break
                if not collision_free:
                    break

            collision_check_array[i] = collision_free
        ######################### GESTIONE OFF ROAD/PEDESTRIAN ################################
        count=0
        count2=0
        collision_check_array_without_off_road=np.zeros(len(paths), dtype=bool)
        if(stopped_at_pedestrian == True or follow_lead_vehicle == True):
            for temp in collision_check_array:
                collision_check_array_without_off_road[count2] = False
                count2 += 1
        else:
            for temp in collision_check_array:
                if (count> len(collision_check_array)-4 and temp == True or (count < 3 and temp == True)):
                    collision_check_array_without_off_road[count]=False
                else:
                    collision_check_array_without_off_road[count]=temp
                count+=1

        return collision_check_array_without_off_road,stopped_at_pedestrian

    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline, and how far away it is from other
    # paths that are in collision. 
    # Disqualifies paths that collide with obstacles from the selection
    # process.
    # collision_check_array contains True at index i if paths[i] is
    # collision-free, otherwise it contains False.
    def select_best_path_index(self, paths, collision_check_array, goal_state):
        """Returns the path index which is best suited for the vehicle to
        traverse.

        Selects a path index which is closest to the center line as well as far
        away from collision paths.

        args:
            paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith path, jth point's t value:
                        paths[i][2][j]
            collision_check_array: A list of boolean values which classifies
                whether the path is collision-free (true), or not (false). The
                ith index in the collision_check_array list corresponds to the
                ith path in the paths list.
            goal_state: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal], unit: [m, m, m/s]
        useful variables:
            self._weight: Weight that is multiplied to the best index score.
        returns:
            best_index: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_index = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                # A lower score implies a more suitable path.
                score = np.sqrt((paths[i][0][-1]-goal_state[0])**2+(paths[i][1][-1]-goal_state[1])**2)

                # Compute the "proximity to other colliding paths" score and
                # add it to the "distance from centerline" score.
                # The exact choice of objective function is up to you.
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            pass

            # Handle the case of colliding paths.
            else:
                score = float('Inf')
                
            # Set the best index to be the path index with the lowest score
            if score < best_score:
                best_score = score
                best_index = i

        return best_index
