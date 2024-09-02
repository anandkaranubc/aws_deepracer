import math

def reward_function(params):
    #############################################################################
    # Read input variables
    #############################################################################
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    heading = params['heading']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    steering_angle = params['steering_angle']
    steps = params['steps']  # Number of steps taken in the episode

    #############################################################################
    # Set the thresholds
    #############################################################################
    SPEED_THRESHOLD = 2.5
    DIRECTION_THRESHOLD = 10.0
    STEERING_THRESHOLD = 15.0
    OPTIMAL_SPEED = 4.0  # Encourage higher optimal speed

    #############################################################################
    # Calculate the reward
    #############################################################################
    reward = 1.0

    # Minimal reward if the car is off the track
    if not all_wheels_on_track:
        return 1e-3

    # Calculate track direction changes
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    next_next_point = waypoints[(closest_waypoints[1] + 1) % len(waypoints)]

    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    next_track_direction = math.atan2(next_next_point[1] - next_point[1], next_next_point[0] - next_point[0])
    track_direction = math.degrees(track_direction)
    next_track_direction = math.degrees(next_track_direction)

    # Adjust reward based on direction difference
    direction_diff = abs(track_direction - heading)
    next_direction_diff = abs(next_track_direction - heading)

    direction_diff = direction_diff if direction_diff <= 180 else 360 - direction_diff
    next_direction_diff = next_direction_diff if next_direction_diff <= 180 else 360 - next_direction_diff

    # Adjust curvature factor based on upcoming direction change
    curvature_factor = next_direction_diff / 45

    # Adjust speed threshold and steering threshold based on curvature
    if curvature_factor > 1:
        SPEED_THRESHOLD *= max(0.5, 1 - curvature_factor)
        STEERING_THRESHOLD += 5 * curvature_factor
    elif curvature_factor < 1 and speed < OPTIMAL_SPEED and next_direction_diff < DIRECTION_THRESHOLD:
        reward *= 1.5  # reward higher speeds on straighter sections

    # Penalize or reward based on steering and speed appropriateness
    if speed > SPEED_THRESHOLD or abs(steering_angle) > STEERING_THRESHOLD:
        reward *= 0.5
    else:
        reward *= 1.0

    # Manage distance from center
    centerline_dist_ratio = distance_from_center / (track_width / 2)
    if centerline_dist_ratio > 0.5:
        reward *= 0.5
    else:
        reward *= 1.0

    return float(reward)
