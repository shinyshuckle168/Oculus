from fov_vect import *

def px_dist(off_nadir, fov_across, fov_along, px_count, cam, rot_mat):
    # calculating pixel distortion and lat/long for a given off nadir angle
    
    # at nadir camera position and rotation matrix
    at_nadir_cam = np.repeat(cam[:,np.newaxis], 1, axis=1)
    at_nadir_rot_mat = np.repeat(rot_mat[:,:,np.newaxis], 1, axis=2)
    
    px_at_nadir = get_area(np.array([[off_nadir[0]]]), np.array([[off_nadir[1]]]), [px_fov_across, px_fov_along], at_nadir_cam, at_nadir_rot_mat)[0][0][0]
    print(px_at_nadir, get_px_area(np.array([off_nadir[0]], dtype=float), np.array([off_nadir[0]], dtype=float), [px_fov_across, px_fov_along], 1.0, at_nadir_cam, at_nadir_rot_mat))

    # off nadir camera position and rotation matrix
    off_nadir_cam = np.repeat(cam[:,np.newaxis], px_count, axis=1)
    off_nadir_rot_mat = np.repeat(rot_mat[:,:,np.newaxis], px_count,axis=2)
    
    # pixel off nadir angles
    off_nadir_across = off_nadir[0]
    off_nadir_across = np.repeat(np.array([off_nadir[0]])[:, np.newaxis], px_count, axis=0)
    off_nadir_along = (off_nadir[1] + (np.arange(px_count)-px_count/2+0.5)*fov_along)[:,np.newaxis]

    px_distortion, px_lat_long = get_area(off_nadir_across, off_nadir_along, [fov_across, fov_along], off_nadir_cam, off_nadir_rot_mat)
    px_distortion = (px_distortion - px_at_nadir)/px_at_nadir*100

    px_distortion2 = get_px_area(np.array([off_nadir[0]]), np.array([off_nadir[1]]), [fov_across, fov_along], px_count, np.repeat(cam[:,np.newaxis], 1, axis=1), np.repeat(rot_mat[:,:,np.newaxis], 1,axis=2))
    px_distortion2 = 100*(px_distortion2 - px_at_nadir)/px_at_nadir

    return px_distortion2, px_lat_long

# number of pixels in a scanline
# assuming even px_count
px_count = 640

# camera position and rotation matrix
cam = np.array((0, 0, (h+r)),dtype=np.int64)
rot_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])

# scanline off nadir angles and pixel field of views
off_nadir = [30*math.pi/180, 30*math.pi/180]
px_fov_across = fov_across
px_fov_along = fov_along/640

px_distortion, px_lat_long = px_dist(off_nadir, px_fov_across, px_fov_along, px_count, cam, rot_mat)

fig = plt.figure()
area_plot = fig.add_subplot(111)

area_plot.set_title('Pixel Area Distortion\n Off Nadir Angle: {} deg across, {} deg along'.format(round(off_nadir[0]*180/math.pi,2), round(off_nadir[0]*180/math.pi,2)))
title = area_plot.title
title.set_weight('bold')
area_plot.set_xlabel('Pixel')
area_plot.set_ylabel('Area Difference Relative to Center Pixel (%)')
area_plot.grid('on')
area_display = area_plot.scatter(np.linspace(0,px_distortion.shape[1]-1,px_distortion.shape[1]), px_distortion[0])

fig2 = plt.figure()
pos_plot = fig2.add_subplot(111)

pos_plot.set_title('Pixel Latitude and Longitude\n Off Nadir Angle: {} deg across, {} deg along'.format(round(off_nadir[0]*180/math.pi,2), round(off_nadir[0]*180/math.pi,2)))
title = pos_plot.title
title.set_weight('bold')
pos_plot.set_xlabel('Latitude (degrees)')
pos_plot.set_ylabel('Longitude (degrees)')
pos_plot.grid('on')
pos_display = pos_plot.scatter(px_lat_long[0,:,0].flatten(), px_lat_long[1,:,0].flatten())

plt.show()