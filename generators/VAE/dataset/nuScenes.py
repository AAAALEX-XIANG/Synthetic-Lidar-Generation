from math import ceil
import os
import torch
from random import random, sample
from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.geometry_utils import BoxVisibility

classes = {'human.pedestrian.adult': 0,
            # 'human.pedestrian.child': 0, void class
            'human.pedestrian.wheelchair': 0,
            'human.pedestrian.stroller': 0,
            'human.pedestrian.personal_mobility': 0,
            'human.pedestrian.police_officer': 0,
            'human.pedestrian.construction_worker': 0,
            'vehicle.car': 1,
            # 'vehicle.emergency.ambulance': 1, void class
            'vehicle.emergency.police': 1,
            'vehicle.motorcycle': 2,
            'vehicle.bicycle': 2,
            'vehicle.bus.bendy': 3,
            'vehicle.bus.rigid': 3,
            'vehicle.truck': 4,
            'vehicle.construction': 5,
            'vehicle.trailer': 6,
            'movable_object.barrier': 7,
            'movable_object.trafficcone': 7,
            'movable_object.pushable_pullable': 7,
            'movable_object.debris': 7,
            # 'animal': 10, void class
            'static_object.bicycle_rack': 8,
        }

class NuScenesData(Dataset):
    def __init__(self, root_dir, points):
        super(NuScenesData, self).__init__()
        self.sample = os.listdir(root_dir)
        self.path = root_dir
        self.cat = classes
        self.classes = list(set(self.cat.values()))
        self.num_points = points
        
    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, index):
        data = torch.load(f'{self.path}/{self.sample[index]}')
        points = torch.from_numpy(data['points'])
        points_volume = points.shape[1]
        labels = torch.tensor(classes[data['semantic_label']])
        #Normalize the point clouds
        ## Get the centroid of point clouds
        pts_centroid = torch.mean(points, axis = 1)
        # Reset the object's coordinates to the origin
        points = torch.sub(points, pts_centroid[:,None])
        #Normalize to [-0.5, 0.5]
        m = torch.max(torch.sqrt(torch.sum(points**2, axis = 0)))
        points = points/(2*m)
        points = points.transpose(0,1)
        if points_volume > self.num_points:
            sample_points = torch.Tensor(sample(points.tolist(),self.num_points))
            sample_points = sample_points.transpose(0,1)
        else:
            replicates = ceil(self.num_points/points_volume)
            sample_points = points.repeat(replicates,1)
            sample_points = sample_points[:self.num_points]
            sample_points = sample_points.transpose(0,1)
        return sample_points, labels

