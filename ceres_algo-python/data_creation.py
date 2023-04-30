import numpy as np
import cv2
import pickle
import random

def generate_3d_points(N):
    points = np.random.rand(N, 3)*100000
    points = np.round(points,0)
    return points

def create_cameras(K):
    cameras = []
    fx = 1500.0
    fy = 1500.0
    cx = 500
    cy = 500
    delta = 1000
    trajectory_y = np.linspace(0,100000,10) + random.randint(-delta, delta)
    trajectory_x = 0.00004*(trajectory_y-50000)**2 + random.randint(-delta, delta)
    trajectory_z = np.ones(trajectory_x.shape) * 45000
    trajectory = np.vstack((trajectory_x, trajectory_y,trajectory_z)).T
    for i in range(K): 
        #rvec = np.random.rand(3)
        #tvec = np.random.rand(3)
        if i == 0:
            tangent_vector = trajectory[i+1] - trajectory[i]
        elif i == K-1:
            tangent_vector = trajectory[i] - trajectory[i-1]
        else:
            tangent_vector = trajectory[i+1] - trajectory[i-1]
            
        tvec = trajectory[i]
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        #cameras.append((rvec, tvec, camera_matrix))
        cameras.append((tangent_vector, tvec, camera_matrix))
        #print('tvec:',tvec,'tang:',tangent_vector)
    return cameras

def write_points_and_cameras(points_file, cameras_file, points, cameras):
    with open(points_file, 'wb') as f:
        pickle.dump(points, f)
    with open(cameras_file, 'wb') as f:
        pickle.dump(cameras, f)

def transform_points_to_camera(points, camera):
    rvec, tvec, camera_matrix = camera
    R, _ = cv2.Rodrigues(rvec)
    T = np.array(tvec).reshape(3, 1)
    points_camera = np.dot(np.linalg.inv(R), (points - T.T).T).T
    return points_camera
# поменять create_cameras rvec, tvec: локализовать фичи с нескольких камер, чтобы фича была видна на n-1 камер, точки около диагонали



def transform_points_to_world(points_camera, camera):
    rvec, tvec, _ = camera
    R, _ = cv2.Rodrigues(rvec)
    T = np.array(tvec).reshape(3, 1)
    points_world = np.dot(R, points_camera.T).T + T.T
    return points_world

def project_points(points, cameras):
    projections = []
    cam_points_3d = []
    visible = []
    visible_on_2_projections = []
    
    for camera in cameras:
        single_cam_points_3d = transform_points_to_camera(points,camera)
        cam_points_3d.append(single_cam_points_3d)
        rvec, tvec, camera_matrix = camera
        projected_points, _ = cv2.projectPoints(single_cam_points_3d, rvec, tvec, camera_matrix, None)
        projected_points = projected_points.reshape(-1, 2)
        projections.append(projected_points)
    
    #projections = np.round(projections,1)
      
    for i, camera in enumerate(cameras):
        single_cam_visible = []
        for j in range(0,len(points)):
            if 0 <= projections[i][j][0] < width and 0 <= projections[i][j][1] < height and cam_points_3d[i][:,2][j]  > 0:
                single_cam_visible.append(True)
            else:
                single_cam_visible.append(False) 
        visible.append(single_cam_visible)
        
    for i in range(0,len(cameras)-1):
        for j in range(i+1,len(cameras)): 
            if j>i:
                for k in range(0,len(points)):
                    if visible[j][k] == visible[i][k]:
                        result = (i, projections[i][k][0], projections[i][k][1], j, projections[j][k][0], projections[j][k][1])
                        visible_on_2_projections.append(result)
                    
    return visible_on_2_projections

def write_projections(projections_file, projections):
    with open(projections_file, 'wb') as f:
        pickle.dump(projections, f)

N = 1000
K = 10
width, height = (1000,1000)
points = generate_3d_points(N)

cameras = create_cameras(K)
write_points_and_cameras('points.pkl', 'cameras.pkl', points, cameras)
projections = project_points(points, cameras)
write_projections('projections.pkl', projections)
print(len(projections))
print(projections[0])
print(cameras[0])



