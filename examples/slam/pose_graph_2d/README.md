Pose Graph 2D
----------------

A pose graph optimization problem falls into the class of problems of
Simultaneous Localization and Mapping (SLAM). The SLAM problem consists of
building a map of an unknown environment while simultaneously localizing against
this map. The main difficulty of this problem stems from not having any
additional external aiding information such as GPS. SLAM has been considered one
of the fundamental challenges of robotics. 

This package defines the necessary Ceres cost functions needed to model the pose
graph optimization problem as well as a binary to build and solve the problem.

Running
-----------
This package includes an executable `pose_graph_2d` that will read a problem
definition file. This executable can work with any 2D problem definition that
uses the g2o format. It would be relatively straightforward to implement a new
reader for a different format such as toro or others. `pose_graph_2d` will print
the Ceres solver full summary and then output to disk the original and optimized
poses (`poses_original.txt` and `poses_optimized.txt`, respectively) of the
robot in the following format:

```
pose_id x y yaw_radians
pose_id x y yaw_radians
pose_id x y yaw_radians
...
```

where `pose_id` is the corresponding integer ID from the file definition. Note,
the file will be sorted in ascending order for the `pose_id`.

The executable `solve_pose_graph_2d` expects the first argument to be the path
to the problem definition. To run the executable,

```
/path/to/bin/pose_graph_2d /path/to/dataset/dataset.g2o
```

where this assumes the install directory is located in the repository.

A python script is provided to visualize the resulting output files.
```
/path/to/repo/robotics/slam/pose_graph_2d/plot_results.py --poses_optimized_filename ./poses_optimized.txt --poses_original_filename ./poses_original.txt
```
