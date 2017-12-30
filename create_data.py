import numpy as np
import struct
from matplotlib import pyplot as plt

file = open('./trainimg', 'rb')
index = 0
buffer = file.read()

magic,num,width,height = struct.unpack_from('>IIII', buffer, index)
index += struct.calcsize('>IIII')

print magic, num, width, height
size = num*width*height

data = struct.unpack_from('>'+str(size)+'B', buffer,index)
data = np.reshape(data, [num, width, height])

file.close()

file = open('./trainlabel', 'rb')
index = 0
buffer = file.read()

magic,num = struct.unpack_from('>II', buffer, index)
index += struct.calcsize('>II')

print magic, num
label = struct.unpack_from('>'+str(num)+'B', buffer,index)
label = np.array(label)

np.savez('mnist_data_label', data=data, label=label)
file.close()

file = open('./testimg', 'rb')
index = 0
buffer = file.read()

magic,num,width,height = struct.unpack_from('>IIII', buffer, index)
index += struct.calcsize('>IIII')

print magic, num, width, height
size = num*width*height

data = struct.unpack_from('>'+str(size)+'B', buffer,index)
data = np.reshape(data, [num, width, height])

file.close()

file = open('./testlabel', 'rb')
index = 0
buffer = file.read()

magic,num = struct.unpack_from('>II', buffer, index)
index += struct.calcsize('>II')

print magic, num
label = struct.unpack_from('>'+str(num)+'B', buffer,index)
label = np.array(label)

np.savez('test_data_label', data=data, label=label)
file.close()