"""Constrained multi-start ICP for the close enclosure, with inverse consistency."""
import math
import numpy as np
from lekiwi_base_bringup import scan_matching as sm

original_icp=sm.icp_2d

def inverse(pose):
    x,y,a=pose;c,s=math.cos(a),math.sin(a)
    return (-c*x-s*y,s*x-c*y,-a)

def robust_icp(source,target,initial=(0.,0.,0.),**kwargs):
    candidates=[]
    for degrees in [0.,-10.,10.,-20.,20.,-30.,30.]:
        seed=(initial[0],initial[1],initial[2]+math.radians(degrees))
        try:
            r=original_icp(source,target,initial=seed,max_distances=(.08,.04))
        except (ValueError,np.linalg.LinAlgError):
            continue
        angle=abs(sm.wrap_angle(r.pose[2]-initial[2]))
        shift=math.hypot(r.pose[0]-initial[0],r.pose[1]-initial[1])
        if angle>math.radians(45.) or shift>.10:
            continue
        # Keep the existing overlap and RMS requirements; add observability.
        if r.inlier_ratio<.6 or r.rms>.02 or r.min_eigenvalue<.005:
            continue
        candidates.append(r)
    candidates.sort(key=lambda r:(-r.inlier_ratio,r.rms))
    for r in candidates:
        try:
            backward=original_icp(target,source,initial=inverse(r.pose),max_distances=(.08,.04))
        except (ValueError,np.linalg.LinAlgError):
            continue
        inv=inverse(backward.pose)
        shift=math.hypot(inv[0]-r.pose[0],inv[1]-r.pose[1])
        angle=abs(sm.wrap_angle(inv[2]-r.pose[2]))
        if backward.inlier_ratio>=.6 and backward.rms<=.02 and shift<.015 and angle<math.radians(1.5):
            print('ICP_INVERSE_CHECK',{'forward_inlier':r.inlier_ratio,
                  'reverse_inlier':backward.inlier_ratio,'delta_m':shift,
                  'delta_deg':math.degrees(angle)},flush=True)
            return r
    raise ValueError('No observable, bidirectionally consistent ICP solution; do not apply radii')
