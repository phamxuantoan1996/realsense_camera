# import numpy

# a = numpy.array([[1,0,3,0,0,5,0],[1,0,3,0,0,5,0]])
# mask = a == 5
# print(mask)

# b = numpy.ma.masked_array(a,mask)

# print(b)

# minial = numpy.min(b[numpy.nonzero(b)])
# print(minial)


import numpy as np

def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
  original_size = np.array(original_size)
  new_size = np.array(new_size)
  original_coordinate = np.array(original_coordinate)
  xy = original_coordinate/(original_size/new_size)
  x, y = int(xy[0]), int(xy[1])
  return (x, y)

output = new_coordinates_after_resize_img((1080,720), (244,244), (102, 34)) # just modify this line
print(output) # output: (23, 11)

