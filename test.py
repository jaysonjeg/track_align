n=[5,10,20,50,100]
anat=[.68, .84, .88, .91, .93]
temp=[.81,.91,.87,.88,3]
temp_niter1 = [.79, .91, .94, .93, .95]

import matplotlib.pyplot as plt
import matplotlib

fig,ax=plt.subplots(1)
ax.plot(n,anat,'b-o',markersize=8)
ax.plot(n,temp_niter1,'r-o',markersize=8)
ax.set_xlabel('Number of subjects')
ax.set_ylabel('Classification accuracy')
ax.legend(['Anatomical alignment', 'Functional alignment'])
ax.set_xscale('log')
ax.set_xticks(n)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()
