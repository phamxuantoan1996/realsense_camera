import numpy

a = numpy.array([[1,0,3,0,0,5,0],[1,0,3,0,0,5,0]])
mask = a == 5
print(mask)

b = numpy.ma.masked_array(a,mask)

print(b)

minial = numpy.min(b[numpy.nonzero(b)])
print(minial)
