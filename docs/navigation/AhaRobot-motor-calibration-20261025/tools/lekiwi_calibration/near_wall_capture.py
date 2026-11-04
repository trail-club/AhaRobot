"""Diagnostic capture for a close enclosure; excludes the base footprint, not nearby walls."""
import time
import math
import warnings
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from lekiwi_base_bringup.calibrate_odom import OdomCalibrator
from lekiwi_base_bringup import scan_matching as sm

class NearWallCalibrator(OdomCalibrator):
    def __init__(self):
        self.last_odom_wall=None
        super().__init__()

    def _on_odom(self,msg):
        super()._on_odom(msg)
        self.last_odom_wall=time.monotonic()

    def move(self,vx,vy,wz,seconds):
        if self.odom_pose is None:
            raise RuntimeError('No encoder odometry; refusing motion')
        start=self.odom_pose
        start_yaw=self.odom_yaw_unwrapped
        angular=abs(wz)>0
        speed=abs(wz) if angular else math.hypot(vx,vy)
        goal=speed*seconds
        deadline=time.monotonic()+2*seconds+5
        try:
            while True:
                if self.last_odom_wall is None or time.monotonic()-self.last_odom_wall>.3:
                    raise RuntimeError('Encoder odometry stale; stopping')
                dx,dy,_=sm.relative_pose(start,self.odom_pose)
                distance=math.hypot(dx,dy)
                progress=(math.copysign(1,wz)*(self.odom_yaw_unwrapped-start_yaw)
                          if angular else (dx*vx+dy*vy)/speed)
                tolerance=math.radians(.25) if angular else .001
                if progress>=goal-tolerance:
                    break
                if time.monotonic()>deadline:
                    raise RuntimeError('Encoder motion target timed out')
                if distance>(.06 if angular else goal*1.5+.015):
                    raise RuntimeError('Unexpected encoder displacement; stopping')
                cmd=Twist()
                if angular:
                    cmd.angular.z=math.copysign(min(speed,max(.08,1.5*(goal-progress))),wz)
                else:
                    v=min(speed,max(.02,2*(goal-progress)))
                    cmd.linear.x=v*vx/speed;cmd.linear.y=v*vy/speed
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self,timeout_sec=.02)
        finally:
            self.stop()

    def capture(self):
        self.stop()
        self.spin_for(float(self.param('settle_s')))
        self.scans=[]
        self.collecting=True
        need=int(self.param('scans_per_capture'))
        deadline=time.monotonic()+15.
        while len(self.scans)<need and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.05)
        self.collecting=False
        if len(self.scans)<need:
            raise RuntimeError(f'Only {len(self.scans)}/{need} scans')
        first=self.scans[0]
        if any(len(s.ranges)!=len(first.ranges) for s in self.scans):
            raise RuntimeError('Scan geometry changed during capture')
        sensor_pose=self.sensor_pose(first.header.frame_id)
        raw=np.asarray([s.ranges for s in self.scans],dtype=float)
        raw[~np.isfinite(raw)]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            ranges=np.nanmedian(raw,axis=0)
            mad=np.nanmedian(np.abs(raw-ranges),axis=0)
        # A point must be observed in at least 70% of scans and remain stable.
        good=(np.isfinite(raw).mean(axis=0)>=.7)&(mad<=.012)
        ranges[~good]=np.inf
        points=sm.scan_to_points(ranges,first.angle_min,first.angle_increment,
                                 .10,float(self.param('max_range')),sensor_pose)
        # Base coordinates: the circular plate and wheels lie inside this exclusion disk.
        points=points[np.linalg.norm(points,axis=1)>.22]
        cloud=sm.voxel_downsample(points,float(self.param('voxel')))
        if len(cloud)<30:
            raise RuntimeError(f'Only {len(cloud)} stable exterior points')
        print('CAPTURE_QUALITY',{'raw_beams':len(first.ranges),'stable_exterior':len(cloud),
              'base_range_min':float(np.linalg.norm(cloud,axis=1).min()),
              'base_range_max':float(np.linalg.norm(cloud,axis=1).max())},flush=True)
        return cloud,self.odom_pose,self.odom_yaw_unwrapped
