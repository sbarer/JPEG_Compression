import inspect, os
#print inspect.getfile(inspect.currentframe()) # script filename (usually with path)
current_dir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
image_subdir = "src/assets/images"

compression_val = str(5)
file_name = 'filename' + compression_val + '.jpg'

image_dir = os.path.join(current_dir, image_subdir, file_name)
print(image_dir)
#GET SIZE OF IMAGE IN BYTES
filesize = os.path.getsize('pupper.jpg')
filesize_kb = float(filesize)/float(1000)
print(filesize_kb)
## SAVE Image into file path
# image.save(image_dir) 
