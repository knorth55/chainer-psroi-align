import argparse

import chainer
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.utils import mask_to_bbox
from chainercv.utils import read_image
from chainercv.visualizations.colormap import voc_colormap
from chainercv.visualizations import vis_bbox
from chainercv.visualizations import vis_instance_segmentation
import matplotlib.pyplot as plt

from psroi_align.links.model import FCISPSROIAlignResNet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='coco')
    parser.add_argument('image')
    args = parser.parse_args()

    proposal_creator_params = FCISPSROIAlignResNet101.proposal_creator_params
    proposal_creator_params['min_size'] = 2
    model = FCISPSROIAlignResNet101(
        n_fg_class=len(coco_instance_segmentation_label_names),
        min_size=800, max_size=1333,
        anchor_scales=(2, 4, 8, 16, 32),
        pretrained_model=args.pretrained_model,
        proposal_creator_params=proposal_creator_params)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = read_image(args.image, color=True)

    masks, labels, scores = model.predict([img])
    mask, label, score = masks[0], labels[0], scores[0]
    bbox = mask_to_bbox(mask)
    colors = voc_colormap(list(range(1, len(mask) + 1)))
    ax = vis_bbox(
        img, bbox, instance_colors=colors, alpha=0.5, linewidth=1.5)
    vis_instance_segmentation(
        None, mask, label, score,
        label_names=coco_instance_segmentation_label_names,
        instance_colors=colors, alpha=0.7, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
