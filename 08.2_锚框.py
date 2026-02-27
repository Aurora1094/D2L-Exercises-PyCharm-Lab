import torch
from d2l import torch as d2l
from d2l.torch import show_bboxes

torch.set_printoptions(2)

def multibox_prior(data,sizes,ratios):
    # 获取输入特征图的高和宽
    in_height,in_width = data.shape[-2:]
    device=data.device
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    # 理论上每个像素可以生成 nxm 个组合。但这样会导致生成的锚框数量爆炸
    # SSD 论文采取了一个折中方案：
    # A:使用所有的 $n$ 个 sizes，但宽高比固定使用 第一个 ratio
    # B：使用第一个 size ($s_1$)，但宽高比使用剩下的 $m-1$ 个 ratios
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 计算中心点坐标
    offset_h=0.5
    offset_w=0.5
    steps_h=1.0/in_height
    steps_w=1.0/in_width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w

    # 生成网格坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
        * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 生成偏移量
    # 将 (-w/2, -h/2, w/2, h/2) 堆叠起来，并重复到每一个像素点上
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
        in_height * in_width, 1) / 2

    # 把中心点坐标 (x, y, x, y) 复制，每个点复制出 boxes_per_pixel 个
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 中心点坐标 + 偏移量 = 最终坐标
    output = out_grid + anchor_manipulations
    # 增加一个 Batch 维度，形状变为 (1, 锚框总数, 4)
    return output.unsqueeze(0)

img=d2l.plt.imread('./img/catdog.jpg')
h,w=img.shape[:2]
# 每个像素点都有n+m-1的可能性锚框
# 宽度 ($w$)：$size \times \sqrt{ratio}$
# 高度 ($h$)：$size / \sqrt{ratio}$
print(h,w)
X=torch.rand(size=(1,3,h,w))
Y=multibox_prior(X,[0.75,0.5,0.25],[1,2,0.5])
print(Y.shape)

# 访问某一像素点的第一个锚框(角落坐标)
boxes=Y.reshape(h,w,5,4)
print(boxes[100,100,0,:])

d2l.set_figsize((10,10))
# 创建一个形状为 (4,) 的张量，包含图片的 (宽, 高, 宽, 高)
# 因为 multibox_prior 输出的坐标是 0 到 1 之间 的归一化数值（如 $0.5$）
# 而绘图函数需要的是实际像素坐标（如 $158 \times 0.5 = 79$ 像素）。相乘即可完成还原。
# bbox_scale用于还原
bbox_scale=torch.tensor((w,h,w,h))
fig=d2l.plt.imshow(img)
show_bboxes(fig.axes,boxes[100,100,:,:]*bbox_scale,['s=0.75,r=1','s=0.5,r=1','s=0.25,r=1','s=0.75,r=2','s=0.75,r=0.5'])
d2l.plt.show()

# 计算IoU
def box_iou(box1,box2):
    # box 的格式是 (xmin, ymin, xmax, ymax)
    box_area=lambda box:((box[:,2]-box[:,0])*((box[:,3]-box[:,1])))
    area1=box_area(box1)
    area2=box_area(box2)

    # 交集左上角：取两个框左上角坐标的最大值 (max)
    inter_upperlefts = torch.max(box1[:, None, :2], box2[:, :2])
    # 交集右下角：取两个框右下角坐标的最小值 (min)
    inter_lowerrights = torch.min(box1[:, None, 2:], box2[:, 2:])

    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = area1[:, None] + area2 - inter_areas

    return inter_areas / union_areas

# 将真实边界框分配给锚框
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    # 获取生成的锚框数量和实际物体的数量
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算每一对“锚框-真实框”的重合度
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)

    # 对每一个锚框，找出它与哪一个真实物体最像
    # 并记录下最高 IoU 值（max_ious）和那个物体的索引（indices）
    max_ious, indices = torch.max(jaccard, dim=1)
    # iou_threshold：通常是 0.5。
    # 这一步的意思是：只要你跟某个物体的重合度超过 0.5，你就被初步指派去负责那个物体。
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 找出这些合格锚框所匹配的那个真实物体的编号
        # ax_ious >= iou_threshold会生成一个由 True 和 False 组成的向量
        # True 的位置：保留该位置在 indices 里的值
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # 清空行/列
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)

    for _ in range(num_gt_boxes):
        # 计算真实坐标
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # 绑定
        anchors_bbox_map[anc_idx] = box_idx
        # 该物体已经分配过了，抹除这一列
        jaccard[:, box_idx] = col_discard
        # 该锚框已经用过了，抹除这一行
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map

# 锚框偏移量转换
# 神经网络直接预测坐标（如 0.1, 0.2）很难收敛
# 因此通常让它预测锚框中心点和宽高的偏移比例
# (角落坐标不容易收敛，使用中心坐标)
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)

    # 计算中心点偏移
    # 乘 10 是为了放大损失，让模型对位置更敏感
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    # 计算宽高偏移
    # 乘 5 同样是是为了放大损失，使用 log 是为了让数值对缩放比例更线性
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    # 合并为 (N, 4) 的偏移量张量
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # 取出当前图片的真实标签
        label = labels[i, :, :]

        # 调用 assign_anchor_to_bbox，得到映射表
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        # 掩码(背景框不计算坐标损失)[圈中物体的锚框，Mask 的值是 1,背景锚框，Mask 的值是 0]
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 初始化类别和分配的真实框坐标
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        # 把那些跟物体对得上的锚框找出来，把它们对应的真实物体的坐标填进 assigned_bb 这个大表里。
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 填充：找出非背景的锚框索引
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]

        # 类别标签 +1：通常 0 是背景，1、2... 是物体类别
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]

        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

# 人工标注
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
# 算法待生成的框（手动生成，和自动那个无关（m+n-1）*pixel）
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()

labels=multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
# return (bbox_offset, bbox_mask, class_labels)
print(labels[2])

# 使用nms预测边界
def nms(boxes, scores, iou_threshold):
    # 把所有框按得分从高到低排序
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 找出那些重叠度较低的框(如果高，则可能预测同一物体，所以不再进入循环)
        # ！！！inds是列表
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


# 把这些偏移量应用到原始锚框上，还原出它们在图上到底指着哪
def offset_inverse(anchors, offset_preds):

    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox



# copy 没完全懂
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景（-1）
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)


anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率


output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

d2l.plt.show()