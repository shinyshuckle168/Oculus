import math, matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Geod, Proj, transform

def get_intersections(across, along, cam, rot_mat):
    # finds intersection of a ray from the camera with the earth  
    
    x, y, z = np.tan(along), np.tan(across), np.ones((across.shape[0],across.shape[1]))

    # unit vector from cam to point on earth 
    u = np.stack((x,y,-1*z))
    u = u/np.linalg.norm(u, axis=0)
    u = np.einsum('ijk,jkb->ikb', rot_mat, u)

    # solving quadratic formula after plugging equation of line into the matrix representation of an ellipsoid
    a = np.einsum('jib,jib->ib', u, np.einsum('i,ijk->ijk',A_diag,u))
    b = 2*np.einsum('ji,jib->ib', cam, np.einsum('i,ijk->ijk',A_diag,u))
    c = np.repeat((np.einsum('ji,ij->i', cam, A_diag*cam.T)-1)[:, np.newaxis], u.shape[2], axis=1)
    d = np.divide(-b-(b**2-4*a*c)**0.5,2*a)

    return np.repeat(cam[:, :, np.newaxis], u.shape[2], axis=2) + d*u

def get_vertices(across, along, fov, cam, rot_mat):
    #finds the coordinates of the corners of the scan line 
    
    # off nadir angles of vertices
    x0_across = across + fov[0]/2
    x0_along =  along + fov[1]/2
    x1_across = across + fov[0]/2
    x1_along =  along - fov[1]/2
    x2_across = across - fov[0]/2
    x2_along =  along - fov[1]/2
    x3_across = across - fov[0]/2
    x3_along =  along + fov[1]/2

    # across list
    vertices_across = np.empty((4*x0_across.shape[0],x0_across.shape[1]))
    vertices_across[0::4,:] = x0_across
    vertices_across[1::4,:] = x1_across
    vertices_across[2::4,:] = x2_across
    vertices_across[3::4,:] = x3_across

    # along list
    vertices_along = np.empty((4*x0_along.shape[0],x0_along.shape[1]))
    vertices_along[0::4,:] = x0_along
    vertices_along[1::4,:] = x1_along
    vertices_along[2::4,:] = x2_along
    vertices_along[3::4,:] = x3_along

    return get_intersections(vertices_across, vertices_along, cam, rot_mat)

def get_lat_long(vertices, rot_mat):
    # computes longitude and latitude of a point with coordinates wrt camera

    # transforms points from wrt camera to wrt earth
    vertices_prime = np.einsum('ijk,jkb->ikb', rot_mat, vertices)

    # converts cartesian coordinates to latitude and longitude
    lat,long,alt = transform(INPROJ,OUTPROJ,vertices_prime[0].flatten(),vertices_prime[1].flatten(),vertices_prime[2].flatten())
    lat = lat.reshape(vertices_prime.shape[1],vertices_prime.shape[2])
    long = long.reshape(vertices_prime.shape[1],vertices_prime.shape[2])

    return np.stack((lat,long))

def get_area(across, along, fov, cam, rot_mat):
    # computes area of the scanline from the latitude and longitude of its vertices

    # extends rot_mat and cam since there are 4 vertices per angle
    rot_mat_new = np.empty([3,3,4*rot_mat.shape[2]])
    rot_mat_new[:,:,0::4] = rot_mat
    rot_mat_new[:,:,1::4] = rot_mat
    rot_mat_new[:,:,2::4] = rot_mat
    rot_mat_new[:,:,3::4] = rot_mat

    cam_new = np.empty([3,4*cam.shape[1]])
    cam_new[:,0::4] = cam
    cam_new[:,1::4] = cam
    cam_new[:,2::4] = cam
    cam_new[:,3::4] = cam

    # calculates vertice positions
    vertices = get_vertices(across, along, fov, cam_new, rot_mat_new)

    # calculates latitude and longitude
    lat_long = get_lat_long(vertices, rot_mat_new)
    
    scanline_area = np.empty((along.shape[0],along.shape[1]))

    for i in range(scanline_area.shape[0]):
        for j in range(scanline_area.shape[1]):
            scanline_area[i,j] = GEOD.polygon_area_perimeter(lat_long[1,i*4:i*4+4,j], lat_long[0,i*4:i*4+4,j])[0]    

    return scanline_area, lat_long

def get_px_area(across, along, fov, px_count, cam, rot_mat): 
    # off nadir angles for each pixel
    px_along = np.repeat(along[:,np.newaxis], px_count, axis=1)
    px_across = np.repeat(across[:,np.newaxis], px_count, axis=1)
    px_across += ((np.arange(px_count)-px_count/2+0.5)*fov[1])[np.newaxis,:]

    # calculates pixel areas using get_area
    return get_area(px_along, px_across, fov, cam, rot_mat)[0]


global A_diag, GEOD, INPROJ, OUTPROJ

r = 6378000

GEOD = Geod('+a=6378137 +f=0.0033528106647475126')

a = 6378137.0 
b = 6356752.314245
A_diag = np.array([1/a**2,1/a**2,1/b**2],dtype=np.float64)

INPROJ = Proj('epsg:4978')
OUTPROJ = Proj('epsg:4326')

l = 52800
w = 82.5

# finding FOVs
h = 500000
phi = math.asin(l/2/r)
y = l/2/math.sin((math.pi-phi)/2)
x = (y**2 - (l/2)**2)**0.5 
h_to_chord = h + x
 
# wrt cam
cam_to_pnt_A = np.array((-l/2, -w/2, -h_to_chord))
cam_to_pnt_B = np.array((l/2, -w/2, -h_to_chord))
cam_to_pnt_C = np.array((l/2, w/2, -h_to_chord))

fov_across = math.acos((np.dot(cam_to_pnt_B, cam_to_pnt_C))/(np.linalg.norm(cam_to_pnt_A)**2))   
fov_along = math.acos((np.dot(cam_to_pnt_A, cam_to_pnt_B))/(np.linalg.norm(cam_to_pnt_A)**2))