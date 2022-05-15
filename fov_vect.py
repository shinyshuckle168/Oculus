import math, matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyproj import Geod, Proj, transform
from scipy.spatial.transform import Rotation as R

def get_intersections(across, along, cam, rot_mat):
    '''
    Finds the intersection of rays from the camera with the earth.  
    
    Given the off nadir angles of rays and camera parameters, uses the quadratic formula to find the intersection of the rays with an ellipsoid modelling the Earth.
    
    Parameters
    ----------
    across : ndarray
        4*N by M array containing the components of the off nadir angle (radians) in the across track direction. Each element represents a different vertex. Rows are snapshots in time; each is associated with a different camera position and orientation.
    along : ndarray
        4*N by M array containing the components of the off nadir angle (radians) in the along track direction. Each element represents a different vertex. Rows are snapshots in time; each is associated with a different camera position and orientation.
    cam : ndarray
        3 by 4*N array containing ECEF coordinates (meters) of the camera. 
    rot_mat : ndarray
        3 by 3 by 4*N array containing the orientation of the camera with respect to the ECEF frame in the form of normalized rotation matrices. By convention, the second column is the vector in the direction of the instantaneous velocity, the third column is the vector along the optical axis pointing towards the scanline, and the first column is the cross product of the second and third columns.

    Returns
    -------
    ndarray
        3 by 4*N by M array of the vertices of each scanline/pixel in ECEF coordinates (meters).
    
    Notes
    -----
    `rot_mat` and `cam` are extended in `get_area` so that entries exist for each vertex and not just for each scanline/pixel.
    Uses WGS84 for ellipsoid model and ECEF coordinates.
    '''
    x, y, z = np.tan(along), np.tan(across), np.ones((across.shape[0],across.shape[1]))

    # unit vector from cam to point on earth 
    u = np.stack((x,y,-1*z))
    u = u/np.linalg.norm(u, axis=0)
    u = np.einsum('ijk,jkb->ikb', rot_mat, u)

    # solving quadratic formula after plugging equation of line into the matrix representation of an ellipsoid
    a = np.einsum('jib,jib->ib', u, np.einsum('i,ijk->ijk', A_diag,u))
    b = 2*np.einsum('ji,jib->ib', cam, np.einsum('i,ijk->ijk', A_diag,u))
    c = np.repeat((np.einsum('ji,ij->i', cam, A_diag*cam.T)-1)[:, np.newaxis], u.shape[2], axis=1)
    d = np.divide(-b-(b**2-4*a*c)**0.5,2*a)

    return np.repeat(cam[:, :, np.newaxis], u.shape[2], axis=2) + d*u

def get_vertices(across, along, fov, cam, rot_mat):
    '''
    Finds the coordinates of the vertices of the scanlines/pixels.

    Given the off nadir angles and camera parameters, calculates the off nadir angles of each vertex. Then calculates the coordinates of the vertices using `get_intersections`.

    Parameters 
    ----------
    across : ndarray
        N by M array containing the components of the off nadir angle (radians) in the across track direction. Each element represents a different scanline/pixel. Rows are snapshots in time; each is associated with a different camera position and orientation.
    along : ndarray
        N by M array containing the components of the off nadir angle (radians) in the along track direction. Each element represents a different scanline/pixel. Rows are snapshots in time; each is associated with a different camera position and orientation.
    fov : array-like
        Two element array with the field of views (radians) of the scanline/pixel in both the across and along track directions.
    cam : ndarray
        3 by 4*N array containing ECEF coordinates (meters) of the camera. 
    rot_mat : ndarray
        3 by 3 by 4*N array containing the orientation of the camera with respect to the ECEF frame in the form of normalized rotation matrices. By convention, the second column is the vector in the direction of the instantaneous velocity, the third column is the vector along the optical axis pointing towards the scanline, and the first column is the cross product of the second and third columns.

    Returns
    -------
    ndarray
        3 by 4*N by M array of the vertices of each scanline/pixel in ECEF coordinates (meters).

    Notes
    -----
    `rot_mat` and `cam` are extended in `get_area` so that entries exist for each vertex and not just for each scanline/pixel.
    Uses WGS84 for ECEF coordinates.
    '''
    
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


    vertices = get_intersections(vertices_across, vertices_along, cam, rot_mat)

    # transforms points from wrt camera to wrt earth
    return np.einsum('ijk,jkb->ikb', rot_mat, vertices)

def get_lat_long(vertices, rot_mat):
    '''
    Computes latitude and longitude coordinates of the vertices found by `get_vertices`.

    Uses Pyproj to transform ECEF coordinates of the vertices to latitude longitude coordinates.

    Parameters
    ----------
    vertices : ndarray
        3 by 4*N by M array of the vertices of each scanline/pixel in ECEF coordinates (meters).
    rot_mat : ndarray
        3 by 3 by 4*N array containing the orientation of the camera with respect to the ECEF frame in the form of normalized rotation matrices. By convention, the second column is the vector in the direction of the instantaneous velocity, the third column is the vector along the optical axis pointing towards the scanline, and the first column is the cross product of the second and third columns.

    Returns
    -------
    ndarray
        2 by 4*N by M array of the vertices of each scanline/pixel in latitude and longitude coordinates (degrees). 

    Notes 
    -----
    Uses WGS84 for ECEF and latitude/longitude coordinates.
    '''

    # converts cartesian coordinates to latitude and longitude
    lat, long, alt = transform(INPROJ,OUTPROJ,vertices_prime[0].flatten(), vertices_prime[1].flatten(), vertices_prime[2].flatten())
    lat = lat.reshape(vertices_prime.shape[1], vertices_prime.shape[2])
    long = long.reshape(vertices_prime.shape[1], vertices_prime.shape[2])

    return np.stack((lat,long))

def get_area(across, along, fov, cam, rot_mat):
    '''
    Compute projected scanline areas and vertice coordinates.

    Given off nadir angles and camera parameters, first computes the coordinates of the vertices of each scanline/pixel using the functions `get_vertices` and `get_lat_long`. Then uses the Pyproj library to compute the surface area of the Earth bounded by those vertices.

    Parameters
    ----------
    across : ndarray
        N by M array containing the components of the off nadir angle (radians) in the across track direction. Each element represents a different scanline/pixel. Rows are snapshots in time; each is associated with a different camera position and orientation.
    along : ndarray
        N by M array containing the components of the off nadir angle (radians) in the along track direction. Each element represents a different scanline/pixel. Rows are snapshots in time; each is associated with a different camera position and orientation.
    fov : array-like
        Two element array with the field of views (radians) of the scanline or scanline/pixel in both the across and along track directions.
    cam : ndarray
        3 by N array containing ECEF coordinates (meters) of the camera. 
    rot_mat : ndarray
        3 by 3 by N array containing the orientation of the camera with respect to the ECEF frame in the form of normalized rotation matrices. By convention, the second column is the vector in the direction of the instantaneous velocity, the third column is the vector along the optical axis pointing towards the scanline, and the first column is the cross product of the second and third columns.

    Returns 
    -------
    scanline_area : ndarray
        N by M array of the surface area of each projected scanline/pixel.
    lat_long : ndarray
        2 by 4*N by M array of the latitudes and longitudes (degrees) of the vertices of each scanline/pixel. 

    Notes
    -----
    Extends `rot_mat` and `cam` so that they are 3 by 4*N and 3 by 3 by 4*N respectively. This is to assign an entry to each vertex. 
    Uses WGS84 for ECEF and latitude/longitude coordinates.
    '''

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
    
    # computes surface area of earth bounded by each set of vertices
    scanline_area = np.empty((along.shape[0],along.shape[1]))

    for i in range(scanline_area.shape[0]):
        for j in range(scanline_area.shape[1]):
            scanline_area[i,j] = GEOD.polygon_area_perimeter(lat_long[1,i*4:i*4+4,j], lat_long[0,i*4:i*4+4,j])[0]    

    return scanline_area, lat_long

def get_px_area(across, along, fov, px_count, cam, rot_mat): 
    '''
    Returns areas and vertice coordinates of each pixel in a scanline 
    
    Given the scanline off nadir angles and camera parameters, computes the off nadir angles of each pixel and finds their area and vertice coordinates using `get_area`.

    Parameters
    ----------
    across : ndarray
        Array of length N containing the components of the off nadir angle (radians) in the across track direction. Each element represents a different scanline. 
    along : ndarray
        Array of length N containing the components of the off nadir angle (radians) in the along track direction. Each element represents a different scanline. 
    fov : array-like
        Two element array with the field of views (radians) of the scanline or scanline in both the across and along track directions.
    px_count : int 
        The number of pixels in a scanline. 
    cam : ndarray
        3 by N array containing ECEF coordinates (meters) of the camera. 
    rot_mat : ndarray
        3 by 3 by N array containing the orientation of the camera with respect to the ECEF frame in the form of normalized rotation matrices. By convention, the second column is the vector in the direction of the instantaneous velocity, the third column is the vector along the optical axis pointing towards the scanline, and the first column is the cross product of the second and third columns.


    Returns
    -------
    ndarray
        N by `px_count` array of the surface area of each projected pixel.
    ndarray
        2 by 4*N by `px_count` array of the latitudes and longitudes (degrees) of the vertices of each pixel. 

    Notes
    -----
    It is assumed that px_count is even.
    Uses WGS84 for ECEF coordinates.
    '''

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