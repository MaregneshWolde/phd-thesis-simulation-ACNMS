import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd


exposed=np.loadtxt('2.grid_observations/exposed_bedrock.txt')
x=np.loadtxt('xcoord.txt')
y=np.loadtxt('ycoord.txt')

xmax=x.max()
xmin=x.min()
scale=xmax-xmin

ymax=y.max()
ymin=y.min()

x=(x-xmin)/scale
y=(y-ymin)/scale



mpl.figure(1)
co=mpl.contour(x,y,exposed,levels=[0.99])   # 


# Now we check if we extracted the correct data


mpl.figure(2)

for seg in co.allsegs[0]:         #  seq contains a line in the contour plot
     x,y=list(zip(*seg))          # x,y contains the location of a point on the exposed bedrock
     mpl.plot(x,y,color='black')
     
mpl.show()

