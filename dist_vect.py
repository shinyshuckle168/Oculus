import time
from fov_vect import *

# calculating area distortion
start = time.time()

# camera position, rotation matrix, and off nadir angles
n = 100
cam = np.array((0, 0, (h+r)),dtype=np.int64)
rot_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
off_nadir_along_arr = np.linspace(0, 30, n)*math.pi/180
off_nadir_across_arr = np.linspace(0, 30, n)*math.pi/180
off_nadir_along_mesh, off_nadir_across_mesh = np.meshgrid(off_nadir_along_arr, off_nadir_across_arr)

# at nadir area
at_nadir_cam = np.repeat(cam[:, np.newaxis], 1, axis=1)
at_nadir_rot_mat = np.repeat(rot_mat[:, :, np.newaxis], 1, axis=2)
at_nadir_area = get_area(np.array([0*math.pi/180])[:,np.newaxis], np.array([0*math.pi/180])[:,np.newaxis].T, [fov_across, fov_along], at_nadir_cam, at_nadir_rot_mat)[0][0][0]

# off nadir area
off_nadir_cam = np.repeat(cam[:, np.newaxis], n, axis=1)
off_nadir_rot_mat = np.repeat(rot_mat[:, :, np.newaxis], n, axis=2)
distortion_arr = np.empty([n,n])
#distortion_arr = get_area(np.repeat(off_nadir_across_arr[:, np.newaxis], n, axis=1), np.repeat(off_nadir_across_arr[:, np.newaxis], n, axis=1).T, [fov_across, fov_along], off_nadir_cam, off_nadir_rot_mat)[0]

get_area(np.repeat(off_nadir_across_arr[:, np.newaxis], n, axis=1), np.repeat(off_nadir_across_arr[:, np.newaxis], n, axis=1).T, [fov_across, fov_along], off_nadir_cam, off_nadir_rot_mat)[0]

# percentage difference between at nadir and off nadir area
distortion_arr = (distortion_arr-at_nadir_area)/at_nadir_area*100

print("Time taken: "+ str(round(time.time()-start,4)) + "s")
'''
#plotting off nadir angle against percent area difference in scanline
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

distortion_plot = ax.scatter(off_nadir_across_mesh*180/math.pi, off_nadir_along_mesh*180/math.pi, distortion_arr)
ax.set_title('Difference in Scanline Area At Nadir and Off Nadir')
ax.set_xlabel('Across Angle (\N{DEGREE SIGN})')
ax.set_ylabel('Along Angle (\N{DEGREE SIGN})')
ax.set_zlabel('Percent Difference (%)')
plt.show()
'''