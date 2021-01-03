from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def trajectories_distance(path1, path2):
    distance, path = fastdtw(path1, path2, dist=euclidean)
    return distance


class TrajectoryAnalyser:
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.distanceMatrix = {}

    def calculate_trajectories_distance_matrix(self):
        for t1 in self.trajectories.keys():
            for t2 in self.trajectories.keys():
                if t1 == t2:
                    continue
                dist = trajectories_distance(self.trajectories[t1], self.trajectories[t2])
                combined_key = '{:d}/{:d}'.format(t1, t2)
                self.distanceMatrix[combined_key] = dist
