# # LightGlue Demo# In this notebook we match two pairs of images using LightGlue with early stopping and point pruning.
# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
from lightglue.viz2d import save_plot
import torch


torch.set_grad_enabled(False)
par_images = Path("assets/megadepth1500")
# ## Load extractor and matcher module# In this example we use SuperPoint features combined with LightGlue.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)

def get_image_pair(file_path, line_number):
    """
    获取指定行号的图像对，并返回两个图像路径。
    
    :param file_path: 文件路径，包含图片配对列表
    :param line_number: 行号，指定读取的图像对
    :return: 一对图像路径 (image0, image1)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if line_number < 1 or line_number > len(lines):
        raise ValueError("行号超出范围")

    # 获取指定行
    line = lines[line_number - 1].strip()  # line_number从1开始, 但list索引从0开始
    image0, image1 = line.split()
    
    return image0, image1

# subimage0, subimage1 =  get_image_pair('assets/pairs.txt',3)
subimage0, subimage1 =  get_image_pair('assets/pairs.txt',5)

image0 = load_image(par_images / subimage0) #torch.Size([3, 682, 1024])
image1 = load_image(par_images / subimage1) #torch.Size([3, 682, 1024])




""" 提取特征点 """
feats0 = extractor.extract(image0.to(device)) #{'keypoints':[1, 2048, 2],'keypoint_scores':[1, 2048],'descriptors':[1, 2048,256],'image_size':[1, 2]}
feats1 = extractor.extract(image1.to(device))



""" 进行匹配 """
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension



kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


""" 画匹配图 """
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
save_plot('1.png')



""" 画关键点图 """
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
save_plot('2.png')



