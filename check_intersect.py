import numpy as np


def is_intersecting(p1, p2, obstacle):
	# Define the four edges of the rectangle
	bottom_left = (obstacle[0], obstacle[1])
	bottom_right = (obstacle[0] + obstacle[2], obstacle[1])
	top_left = (obstacle[0], obstacle[1] + obstacle[3])
	top_right = (obstacle[0] + obstacle[2], obstacle[1] + obstacle[3])
	
	rectangle_edges = [
		(bottom_left, bottom_right),
		(bottom_right, top_right),
		(top_right, top_left),
		(top_left, bottom_left)
	]
	
	for edge in rectangle_edges:
		if line_intersection(p1, p2, edge[0], edge[1]):
			return True
	
	return False


def line_intersection(line1, line2, line3, line4):
	def ccw(A, B, C):
		return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
	
	A, B = line1, line2
	C, D = line3, line4
	return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# Test
p1 = np.array([6.59,   5.15])
p2 = np.array([4.42, 0.79])
obstacle = np.array([4, 4, 2, 2])
print(is_intersecting(p1, p2, obstacle))
print("Direction: ", p2 - p1)
print("Normalised: ", (p2 - p1)/np.linalg.norm(p2 - p1))
a_max = 1.0
print("Clipped: ", np.clip((p2 - p1)/np.linalg.norm(p2 - p1), -a_max, a_max))
print("With Noise: ", np.random.normal(loc=np.clip((p2 - p1)/np.linalg.norm(p2 - p1), -a_max, a_max), scale=0.0))
