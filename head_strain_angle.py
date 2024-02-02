import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from skspatial.objects import Plane
import os


mesh_path = r"D:\Marc\Simulations\Drone\Simus_lucille\Raymond_35mps\GHBMC_M50-O_v6-0_Nodes.k"
# results_file_path = r"D:\Marc\Tests\Raymond_35.1mps_Serosion_elform2\max_principal_strain_gage_Outer_solid_nodal"
results_file_path = r"D:\Marc\Simulations\Drone\Impactor_size\Results_fixedMassUpper_max_principal_stress_gage_Outer_shell_nodal\14.34mm.txt"
path_save_data = os.path.splitext(results_file_path)[0] + ".npy"  # If not None, saves the data for post-treatment purpose.
impact_center = [-169.006, 85, -692.26]  # Approximate coordinates of the impact center
time_sample_nb = 5  # Number of results analysed at a specific time (evenly spaced bewteen starting and ending time).
sphere_radius = 4  # Radius of the sphere to average the nodal results.
dist = sphere_radius  # Distance between each sample sphere center
preview = False  # Show graphically the first iteration
angles = [i * 45 for i in range(8)]  # First directions in ZX plane
# angles = [45]  # First directions in ZX plane

def extract_nodes_from_keyfile(keyfile_path):
    """
    Extract node ids and coordinates from keyfile
    :param keyfile_path: Path to the keyfile containing the nodes' ids and coordinates.
    :return: Dictionary {node_id: [coord_x, coord_y, coord_z]}
    """
    variable_length = 8  # Character length of variables
    node_dict = {}
    with open(keyfile_path, 'r') as f:
        is_node = False
        while True:
            line = f.readline()
            if not line:
                break

            if "*NODE" in line:
                line = f.readline()
                while line[0] == "$":
                    line = f.readline()
                is_node = True

            if is_node:
                if "*END" in line:
                    break
                node_id = int(line[:variable_length])
                x = float(line[variable_length:3*variable_length])
                y = float(line[3*variable_length:5*variable_length])
                z = float(line[5*variable_length:7*variable_length])
                node_dict[node_id] = {"coord": [x, y, z]}

    return node_dict

def extract_nodalvalue_from_file(file_path):
    """

    Extract nodal value (strain, stress, etc.) from text file.
    :param file_path: Path to the file containing the nodes' ids and value.
    :param time: if not None, returns only the value at the specified time (nearest time available in file).
    :return: Dictionary {time: {node_id: value}}
    """
    variable_length = 10  # Character length of variables
    node_dict_values = {}
    with open(file_path, 'r') as f:
        is_node = False
        while True:
            line = f.readline()
            if not line:
                break

            if "TIME_VALUE" in line:
                time = float(line.split("=")[-1])
                node_dict_values[time] = {}
                line = f.readline()
                while line[0] == "$":
                    line = f.readline()
                is_node = True

            if is_node:
                if "*END" in line:
                    is_node = False
                    continue
                node_id = int(line[:variable_length])
                value = float(line[variable_length:2*variable_length])
                node_dict_values[time][node_id] = value

    return node_dict_values

def closest_node_id(coord_: list, node_coords: dict):
    """
    Find the closest node from the coordinates.
    :param coord_: [x, y, z] coordinates from which to find the closest node
    :param node_coords: Dictionary {node_id: {'coord': [coord_x, coord_y, coord_z]}} from which to find the closest node.
    :return: The closest node's id
    """
    coord = np.array(coord_)
    sq_dists = {node_id: np.sum((np.array(node_coords[node_id]['coord']) - coord)**2, axis=0)
                for node_id in node_coords.keys()}
    return min(sq_dists, key=sq_dists.get)

def node_id_walk(node_id_start, first_direction, node_coords, plane_orthogonal_direction=None, length=10):
    """
    Take a starting node (node_id_start) and walks through given nodes (node_coords) to find the next node at a distance
    of "length" in the direction "first_direction". Continue the walk in the direction defined by the two last nodes. If
    a plane defined by the orthogonal vector "plane_orthogonal_direction" is given, the direction is constrained to this
    plane.
    :param node_id_start: Id of the starting node in "node_coords".
    :param first_direction: 3D vector (x, y, z) defining the first direction to walk to.
    :param node_coords: Dictionary {node_id: {'coord': [x, y, z]}}.
    :param plane_orthogonal_direction: Normal to a plane constraining the direction (optional).
    :param length: Distance between to nodes.
    :return: The node_ids of each node of the walk, the distance between each node and the starting node, coordinate of
    each node OR (if a plane is given) the coordinates of the projection of the node on the plane.
    """
    node_ids = []
    dists = []
    coords_on_plane = []
    node_id0 = node_id_start
    coord0 = np.array(node_coords[node_id0]['coord'])
    direction = np.array(first_direction)
    prev_direction = direction
    node_ids.append(node_id0)
    coords_on_plane.append(coord0)
    dist = 0
    dists.append(dist)
    for i in range(20):
        coord1 = coord0 + direction
        node_id1 = closest_node_id(coord1, node_coords)
        node_ids.append(node_id1)

        coord0 = np.array(node_coords[node_id1]['coord'])
        direction = coord0 - np.array(node_coords[node_id0]['coord'])
        if plane_orthogonal_direction is not None:
            plane = Plane(point=np.array(node_coords[node_id_start]['coord']), normal=plane_orthogonal_direction)
            coord0 = plane.project_point(coord0)
            direction = coord0 - plane.project_point(np.array(node_coords[node_id0]['coord']))
        if np.sum(direction**2) == 0:
            direction = prev_direction
        direction = direction / np.linalg.norm(direction) * length
        dist += np.sqrt(np.sum((coords_on_plane[-1]-coord0)**2))
        dists.append(dist)
        coords_on_plane.append(coord0)
        prev_direction = direction

        node_id0 = node_id1
    return node_ids, dists, coords_on_plane

def nodes_in_sphere(sphere_center, sphere_radius, coords):
    """
    Detect coordinates that are in the sphere of radius "sphere_radius" and center "sphere_center".
    :param sphere_center: Coordinates [x, y, z] of the center of the sphere.
    :param sphere_radius: Radius of the sphere.
    :param coords: List of coordinates [[x, y, z]] to be tested.
    :return: Ids of the coordinates in "coords" that are in the sphere.
    """
    cx, cy, cz = sphere_center
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    inside_sphere_ids = np.where((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < sphere_radius ** 2)[0]
    return inside_sphere_ids

def get_values_along_path(mesh_path, results_file_path, time, impact_center, angle_direction, dist, sphere_radius,
                          plotter=None, scalars=None):
    """
    (Simple description) Walks along the mesh in a specified direction and averages the nodal results at each sampled
    node.
    Get average result values (in "results_file_path") of the mesh in "mesh_path" at each coordinate defined by the
    nearest node from the "impact_center" (starting node) along the path in the direction "angle_direction" (in ZX
    plane) sampled at each distance "dist" of the nodal values for all nodes contained in the sphere of radius
    "sphere_radius" at each sampled node.
    :param mesh_path: Path to the file containing the mesh node ids and coordinates.
    :param results_file_path: Path to the file containing the mesh node ids and nodal values (strain or stress or etc.).
    :param time: Specific time to extract the results.
    :param impact_center: Coordinates of the impact center
    :param angle_direction: Angle at which the sampling of the nodes will start (in ZX plane)
    :param dist: Distance between each sampled node
    :param sphere_radius: Radius of the sphere to average the values.
    :param plotter: if not None, pyvista plotter to attach the mesh's nodes, values, sampled nodes, and plane.
    :return: "vals": averaged values at each sampled nodes. "dists": distance of each sampled node from the starting
    node.
    """
    node_coords = extract_nodes_from_keyfile(mesh_path)
    node_values = extract_nodalvalue_from_file(results_file_path)
    node_values = node_values[time]  # Keep only time of interest
    node_coords = {key: node_coords[key] for key in node_values}  # Keep only the nodes with an extracted value
    impact_node_id = closest_node_id(impact_center, node_coords)  # Get the closest node from the impact center
    first_direction = np.array([np.sin(angle_direction * np.pi / 180), 0, np.cos(angle_direction * np.pi / 180)]) * dist  # First direction defined in function of an angle in the ZX plane.
    try:
        plane_orthogonal_direction = np.cross(first_direction, (0, 1, 0))  # Constraint plane defined by the direction and the Y vector
    except:
        pass
    sample_measure_node_ids, dists, coords_on_plane = node_id_walk(impact_node_id, first_direction, node_coords,
                                                                   plane_orthogonal_direction=plane_orthogonal_direction,
                                                                   length=dist)


    coords = np.array([node_coords[id]['coord'] for id in node_values])
    vals = np.array([node_values[id] for id in node_values])
    sample_vals = []  # Average of strain value for nodes in sphere at each coords_on_plane
    nb_nodes_inside = []
    for sphere_center in coords_on_plane:
        inside_sphere_ids = nodes_in_sphere(sphere_center, sphere_radius, coords)
        sample_vals.append(np.mean(vals[inside_sphere_ids]))
        nb_nodes_inside.append(len(inside_sphere_ids))
    print(f"Mean number of nodes inside sphere : {np.mean(nb_nodes_inside)} (SD: {np.std(nb_nodes_inside)})")

    if plotter is not None:
        mesh = pv.PolyData(coords)
        if scalars is None:
            scalars = vals
        plotter.add_mesh(mesh, scalars=scalars, cmap="jet", point_size=10)  #, clim=[0, 0.04]
        plotter.add_points(np.array([impact_center]))
        plotter.add_points(np.array(
            node_coords[closest_node_id(impact_center, {key: node_coords[key] for key in node_coords})]['coord']),
                      color="r", render_points_as_spheres=True, point_size=10)
        for sphere_center in coords_on_plane:
            sphere = pv.Sphere(radius=sphere_radius, center=sphere_center)
            plotter.add_mesh(sphere, color="y", opacity=0.4)

        # plane = pv.Plane(node_coords[impact_node_id]['coord'], plane_orthogonal_direction, i_size=300, j_size=300)
        # plotter.add_mesh(plane, color='r', opacity=0.3)
        plotter.show_axes_all()

    return sample_vals, dists


# Looking for the time where max value occur
vals = extract_nodalvalue_from_file(results_file_path)
time_at_max = 0
max_val = 0
for time in vals:
    current_max_val = np.max(list(vals[time].values()))
    if max_val < current_max_val:
        max_val = current_max_val
        time_at_max = time
print(f"Time at max value : {time_at_max} s")
vals_times_at_max = list(vals[time_at_max].values())

times = list(extract_nodalvalue_from_file(results_file_path).keys())[:13]
times = [times[i] for i in np.unique(np.linspace(0, len(times) - 1, time_sample_nb).astype(int))] + [time_at_max]
data = {}
for angle in angles:
    data[angle] = {}
    for time in times:
        print(f"Analyzing results at angle {angle}° and time {time} s")
        if preview:
            plotter = pv.Plotter()
        else:
            plotter = None
        sample_vals, dists = get_values_along_path(mesh_path, results_file_path, time, impact_center, angle, dist, sphere_radius, plotter=plotter, scalars=vals_times_at_max)
        data[angle][time] = [sample_vals, dists]

        if preview:
            plotter.show()
            preview = False

if path_save_data is not None:
    np.save(path_save_data, data, allow_pickle=True)

fig, axs = plt.subplots(3, 3)
max_val = 0
max_time = 0
for i, angle in enumerate(data.keys()):
    for j, time in enumerate(data[angle].keys()):
        max_current = np.max(data[angle][time][0])
        if max_val < max_current:
            max_val = max_current
        sample_vals, dists = data[angle][time]
        axs.flatten()[i].plot(dists, sample_vals, label=f"{time:.2f} s")
        axs.flatten()[i].set_title(f"Angle {angle} °")
        axs.flatten()[i].legend()
for ax in axs.flatten():
    ax.set_ylim(0, max_val)

plt.show()
