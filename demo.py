# # LightGlue Demo# In this notebook we match two pairs of images using LightGlue with early stopping and point pruning.
# If we are on colab: this clones the repo and installs the dependencies
from pathlib import Path

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
from lightglue.viz2d import save_plot
import torch

original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

torch.set_grad_enabled(False)
images = Path("assets")
# ## Load extractor and matcher module# In this example we use SuperPoint features combined with LightGlue.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)



""" 加载图像 """
# image0 = load_image(images / "29307281_d7872975e2_o.jpg") #torch.Size([3, 682, 1024])
# image1 = load_image(images / "62689091_76cdd0858b_o.jpg") #torch.Size([3, 682, 1024])

# image0 = load_image(images / "Javeri/frame_287_jpg.jpg") #torch.Size([3, 682, 1024])
# image1 = load_image(images / "Javeri/frame_297_jpg.jpg") #torch.Size([3, 682, 1024])


image0 = load_image(images / "banana/cc_DJI_20240323105917_0001_D.JPG") #torch.Size([3, 682, 1024])
image1 = load_image(images / "banana/cc_DJI_20240323174053_0108_D.JPG") #torch.Size([3, 682, 1024])




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




# """ demo2"""
# image0 = load_image(images / "sacre_coeur1.jpg")
# image1 = load_image(images / "sacre_coeur2.jpg")

# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension

# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]



# """ 画匹配图 """
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
# save_plot('3.png')

# """ 画关键点图 """
# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
# save_plot('4.png')