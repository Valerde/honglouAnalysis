import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_yahei_consolas = FontProperties(fname="YaHei.Consolas.1.11b.ttf")
#%matplotlib inline

x = range(10)
plt.plot(x)
plt.title("中文",
          fontproperties=font_yahei_consolas,
          fontsize=14)
plt.show()