import numpy as np
from scipy.spatial.transform import Rotation as R
from fov_vect import *
import matplotlib.pyplot as plt
from matplotlib import cm

positions = np.load("supernova data/ECI_positions.npy").T
t = np.load("supernova data/timesteps.npy")
quats = np.load("supernova data/rotation_quats.npy")
rot_mat = R.from_quat(quats).as_matrix()
rot_mat = np.transpose(rot_mat, (2,1,0))
windows = np.load("supernova data/imaging_windows.npy", allow_pickle=True)

# splitting data into list of passes 
cam = [None]*(windows.shape[0])
pos_ls = [None]*(windows.shape[0])
t_ls = [None]*(windows.shape[0])
rot_mat_ls = [None]*windows.shape[0]

for i, window in enumerate(windows):
    i_0 = window[0][0]
    i_n = window[0][-1]

    # initial time and position if they are not provided
    if i == 0 and i_0 == 0:
        # uses time step between t[0] and t[1] for initial time
        delta_t = t[1]-t[0]
        # use velocity at t[1]
        v_init = (positions[:,2] - positions[:,0])/(t[2] - t[0])
        # calculates initial position from velocity and time
        pos_init = positions[:,2] - v_init*delta_t
        
        t_ls[0] = np.insert(t[np.arange(i_0, i_n+1)], 0, delta_t, axis=0)
        pos_ls[0] = np.insert(positions[:, np.arange(i_0, i_n+1)], 0, pos_init, axis=1)
    
    # final time and position if they are not provided
    elif i == windows.shape[0]-1 and i_n == positions.shape[1]:
        # uses time step between t[-1] and t[-2] for initial time
        delta_t = t[-1]-t[-2]
        # use velocity at t[-2]
        v_f = (positions[:,-1] - positions[:,-3])/(t[-1] - t[-3])
        # calculates initial position from velocity and time
        pos_f = positions[:,-1] + v_f*delta_t
        
        t_ls[-1] = np.insert(t[np.arange(i_0, i_n+1)], 0, delta_t, axis=0)
        pos_ls[-1] = np.insert(positions[:, np.arange(i_0, i_n+1)], 0, pos_f, axis=1)

    else:
        pos_ls[i] = positions[:, np.arange(i_0-1, i_n+1)]
        t_ls[i] = t[np.arange(i_0-1, i_n+1)]

    # normalizes position vector
    cam[i] = pos_ls[i]*1000 
    pos_ls[i] = -pos_ls[i]/np.linalg.norm(pos_ls[i], axis=0)

    rot_mat_ls[i] = rot_mat[:, :, np.arange(i_0, i_n)]
    rot_mat_ls[i][:,2,:] *= -1

# Hill frame
# radial is just position since it is in ECI frame

# velocity vector
v = [None]*windows.shape[0]
for i in range(len(windows)):
    v[i] = (pos_ls[i][:,2:]-pos_ls[i][:,:-2])/2/(t_ls[i][2:]-t_ls[i][:-2])
    v[i] = v[i]/np.linalg.norm(v[i], axis=0)

# velocity cross radial vector
cross = [None]*windows.shape[0]
for i in range(len(windows)):
    cross[i] = np.cross(v[i], pos_ls[i][:,1:-1], axisa=0, axisb=0).T

# body frame
# along track vector when at nadir
y = [None]*windows.shape[0]
for i in range(len(windows)):
    y[i] = rot_mat_ls[i][:,1,:]

# across track vector when at nadir
x = [None]*windows.shape[0]
for i in range(len(windows)):
    x[i] = rot_mat_ls[i][:,0,:]

# points to scanline
z = [None]*windows.shape[0]
for i in range(len(windows)):
    z[i] = rot_mat_ls[i][:,2,:]

# calculating off nadir angles
# projecting radial to y-z plane; angle between projection and z is along track angle
proj_yz = [None]*windows.shape[0]
along = [None]*windows.shape[0]
for i in range(len(windows)):
    proj_yz[i] = pos_ls[i][:,1:-1] - np.einsum('ij,ij->j', pos_ls[i][:,1:-1], x[i])*x[i]
    along[i] = np.arccos(np.einsum('ij,ij->j', proj_yz[i], z[i])/np.linalg.norm(proj_yz[i], axis=0))

# projecting radial to x-z plane; angle between projection and z is across track angle
proj_xz = [None]*windows.shape[0]
across = [None]*windows.shape[0]
for i in range(len(windows)):
    proj_xz[i] = pos_ls[i][:,1:-1] - np.einsum('ij,ij->j',pos_ls[i][:,1:-1], y[i])*y[i]
    across[i] = np.arccos(np.einsum('ij,ij->j', proj_xz[i], z[i])/np.linalg.norm(proj_xz[i], axis=0))

n=10
hill_mat = np.array([cross[0][:,:n],v[0][:,:n],-1*pos_ls[0][:,1:n+1]])
area = get_px_area(across[0][:n], along[0][:n], [fov_across, fov_along/640], 640, cam[0][:,:n], hill_mat)

area = (area - np.min(area))/(np.max(area) - np.min(area))

fig = plt.figure()
area_plot = fig.add_subplot(111)

psm = area_plot.pcolormesh(area, cmap=cm.get_cmap('viridis', 100), rasterized=True, vmin=0, vmax=1)
fig.colorbar(psm, ax=area_plot)
plt.title('Normalized Area for First Imaging Window')
area_plot.set_yticks(np.arange(10))
area_plot.set_yticklabels(labels=np.round(across[0][:10]*180/math.pi, 4).astype(str).tolist(), va = 'center')
plt.xlabel('Pixel')
plt.ylabel('Across Angle')

plt.show()