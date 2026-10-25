"""Explicitly invoked physical LeKiwi evaluation; records constant-time and target motion separately."""
import sys,time,json,math,traceback
from pathlib import Path
import numpy as np
import rclpy
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
sys.path.insert(0,'/maps')
from near_wall_capture import NearWallCalibrator
from near_wall_matching import robust_icp
from lekiwi_base_bringup import scan_matching as sm
out=Path(f'/maps/odom_final_evaluation_{int(time.time())}');out.mkdir()
report={'steps':[],'captures':[],'trajectory':[],'commands':[],'radii':None}
class Recorder(NearWallCalibrator):
 def _on_odom(self,msg):
  super()._on_odom(msg)
  report['trajectory'].append({'t':time.monotonic(),'pose':self.odom_pose,'yaw':self.odom_yaw_unwrapped})
rclpy.init();node=Recorder();node.set_parameters([Parameter('voxel',value=.005),Parameter('settle_s',value=2.),Parameter('scans_per_capture',value=20)])
def cap(name):
 c,p,y=node.capture();np.save(out/f'{name}.npy',c)
 if p is None:raise RuntimeError('No encoder odometry')
 report['captures'].append({'name':name,'pose':p,'yaw':y});return c,p,y
def clearance(cloud,sign):
 angle=0 if sign>0 else math.pi
 a=np.arctan2(cloud[:,1],cloud[:,0]);d=np.arctan2(np.sin(a-angle),np.cos(a-angle))
 p=cloud[np.abs(d)<math.pi/6];q=p[:,0]*sign
 if len(p)<12 or float(q.min())<.275:raise RuntimeError('Insufficient observed forward/back corridor')
def timed(v,w,seconds):
 start=node.odom_pose;yaw=node.odom_yaw_unwrapped;beg=time.monotonic();sent=[]
 try:
  while time.monotonic()-beg<seconds:
   if node.last_odom_wall is None or time.monotonic()-node.last_odom_wall>.3:raise RuntimeError('Stale encoder odometry')
   dx,dy,_=sm.relative_pose(start,node.odom_pose)
   if math.hypot(dx,dy)>(.06 if w else abs(v)*seconds*1.5+.015):raise RuntimeError('Unexpected displacement')
   if abs(node.odom_yaw_unwrapped-yaw)>(abs(w)*seconds*1.5+.10 if w else .15):raise RuntimeError('Unexpected rotation')
   cmd=Twist();cmd.linear.x=v;cmd.angular.z=w
   now=time.monotonic();node.cmd_pub.publish(cmd);sent.append(now)
   report['commands'].append({'t':now,'vx':v,'vy':0.,'wz':w})
   rclpy.spin_once(node,timeout_sec=.02)
 finally:node.stop()
 return {'duration_requested_s':seconds,'publish_and_stop_duration_s':time.monotonic()-beg,'first_to_last_publish_s':sent[-1]-sent[0] if sent else 0,'publish_count':len(sent),'vx':v,'wz':w}
def compare(name,b,a,meta):
 d=sm.relative_pose(b[1],a[1]);row={'name':name,**meta,'odom':[d[0],d[1],a[2]-b[2]]}
 try:row['lidar']=vars(robust_icp(a[0],b[0],initial=d))
 except Exception as e:row['registration_error']=str(e)
 report['steps'].append(row);print('MEASUREMENT',json.dumps(row),flush=True)
try:
 report['radii']=node.driver_radii();first=current=cap('start')
 for name,sign in [('timed_forward',1),('timed_backward',-1)]:
  clearance(current[0],sign);node.spin_for(.25)
  print('MOVE',name,flush=True);meta=timed(sign*.03,0.,1.)
  after=cap(name);compare(name,current,after,{'mode':'constant_time',**meta});current=after
 for name,sign in [('target_forward',1),('target_backward',-1)]:
  clearance(current[0],sign);node.spin_for(.25);print('MOVE',name,flush=True)
  node.move(sign*.03,0.,0.,1.)
  after=cap(name);compare(name,current,after,{'mode':'encoder_target','target_m':sign*.03});current=after
 for name,sign in [('rotation_left',1),('rotation_right',-1)]:
  if float(np.linalg.norm(current[0],axis=1).min())<.235:raise RuntimeError('Insufficient rotational clearance')
  node.spin_for(.25);print('MOVE',name,flush=True);node.move(0.,0.,sign*.3,2*math.pi/.3)
  after=cap(name);compare(name,current,after,{'mode':'encoder_target','target_deg':sign*360});current=after
 for name,sign in [('timed_rotation_left',1),('timed_rotation_right',-1)]:
  if float(np.linalg.norm(current[0],axis=1).min())<.235:raise RuntimeError('Insufficient rotational clearance')
  node.spin_for(.25);print('MOVE',name,flush=True);meta=timed(0.,sign*.2,2.)
  after=cap(name);compare(name,current,after,{'mode':'constant_time',**meta});current=after
 compare('return_to_start',first,current,{'mode':'net_return'})
except Exception as e:
 report['error']=str(e);traceback.print_exc()
finally:
 node.stop();(out/'report.json').write_text(json.dumps(report,indent=2));node.destroy_node()
 if rclpy.ok():rclpy.shutdown()
 print('SAVED',out,flush=True)
