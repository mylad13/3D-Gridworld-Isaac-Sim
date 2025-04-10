#!/usr/bin/env python3
import os
import yaml
import cv2
import numpy as np
import argparse


import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class MapSaver(Node):
    def __init__(self, namespace: str, map_dir: str, map_filename: str = "map",
                 free_thresh: float = 0.25, occ_thresh: float = 0.65):
        super().__init__(f'persistent_map_saver_{namespace}')
        self.namespace = namespace
        self.map_dir = map_dir
        self.map_filename = map_filename
        self.free_thresh = free_thresh
        self.occ_thresh = occ_thresh
        self.map_received = False

        # Subscribe to the map topic in the given namespace
        topic = f'/{namespace}/map_low'
        self.subscription = self.create_subscription(
            OccupancyGrid,
            topic,
            self.map_callback,
            10
        )
        self.get_logger().info(f"MapSaver started, subscribed to {topic}")

    def map_callback(self, msg: OccupancyGrid):

        self.get_logger().info("Map received; saving map files.")
        width = msg.info.width
        height = msg.info.height

        # Convert the OccupancyGrid data to a numpy array and reshape it.
        # Note: nav2_map_server applies some conversions; below is a basic example.
        data = np.array(msg.data, dtype=np.int8).reshape(height, width)
        # For demonstration, we convert unknown (-1), free (0), and occupied values:
        # Typically, -1 (unknown) is set to gray, free (0) to white, occupied (>=100) to black.
        img = np.zeros((height, width), dtype=np.uint8)
        img[data == -1] = 205   # Unknown
        img[data == 0] = 255    # Free
        img[data >= 100] = 0    # Occupied

        # Save the image as pgm
        pgm_path = os.path.join(self.map_dir, self.map_filename + '.pgm')
        cv2.imwrite(pgm_path, img)
        self.get_logger().info(f"Map image saved to {pgm_path}")

        # Create a YAML file with the required metadata
        yaml_data = {
            'image': self.map_filename + '.pgm',
            'mode': 'trinary',
            'resolution': msg.info.resolution,
            'origin': [msg.info.origin.position.x,
                       msg.info.origin.position.y,
                       msg.info.origin.position.z],
            'negate': 0,
            'occupied_thresh': self.occ_thresh,
            'free_thresh': self.free_thresh
        }
        yaml_path = os.path.join(self.map_dir, self.map_filename + '.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
        self.get_logger().info(f"Map YAML saved to {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a map subscriber node. Saves the robot's map into pgm and yaml files.")
    parser.add_argument("--namespace", type=str, default="robot1", help="The namespace of the robot to subscribe to.")
    parser.add_argument("--map_dir", type=str, default="maps", help="Directory to save the map files.")
    args = parser.parse_args()
    
    rclpy.init(args=None)
    namespace = args.namespace

    map_dir = os.path.join(args.map_dir, namespace)
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    node = MapSaver(namespace=namespace, map_dir=map_dir)
    
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("MapSaver interrupted by user, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

