import matplotlib.pyplot as plt
import numpy as np
import json

# 生成 Viridis colormap 的 256 个颜色点
cmap = plt.get_cmap('viridis')
colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in cmap(np.linspace(0, 1, 256))]

# 保存为 JSON
with open("viridis.json", "w") as f:
    json.dump(colors, f)

