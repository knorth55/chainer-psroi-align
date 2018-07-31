from __future__ import division

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain \
    import _fast_rcnn_loc_loss
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator \
    import AnchorTargetCreator

from psroi_align.links.model.fcis.utils.proposal_target_creator \
    import ProposalTargetCreator


class FCISTrainChain(chainer.Chain):

    """Calculate losses for FCIS and report them.

    This is used to train FCIS in the joint training scheme [#FCISCVPR]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`roi_mask_loss`: The mask loss for the head module.

    .. [#FCISCVPR] Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei. \
    Fully Convolutional Instance-aware Semantic Segmentation. CVPR 2017.

    Args:
        fcis (~chainercv.experimental.links.model.fcis.FCIS):
            A FCIS model for training.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FCISCVPR]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FCISCVPR]_.
        anchor_target_creator: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator: An instantiation of
            :class:`~chainercv.experimental.links.model.fcis.ProposalTargetCreator`.

    """

    def __init__(
            self, fcis,
            rpn_sigma=3.0, roi_sigma=1.0,
            anchor_target_creator=AnchorTargetCreator(),
            proposal_target_creator=ProposalTargetCreator()
    ):

        super(FCISTrainChain, self).__init__()
        with self.init_scope():
            self.fcis = fcis
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.mask_size = self.fcis.head.roi_size

        self.loc_normalize_mean = fcis.loc_normalize_mean
        self.loc_normalize_std = fcis.loc_normalize_std

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

    def __call__(self, imgs, masks, labels, bboxes, scales):
        """Forward FCIS and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.
        * :math:`H` is the image height.
        * :math:`W` is the image width.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            masks (~chainer.Variable): A batch of masks.
                Its shape is :math:`(N, R, H, W)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            scales (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(masks, chainer.Variable):
            masks = masks.array
        if isinstance(labels, chainer.Variable):
            labels = labels.array
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.array
        if isinstance(scales, chainer.Variable):
            scales = scales.array
        scales = cuda.to_cpu(scales)

        batch_size, _, H, W = imgs.shape
        img_size = (H, W)
        assert img_size == masks.shape[2:]

        if any(len(b) == 0 for b in bboxes):
            return chainer.Variable(self.xp.array(0, dtype=np.float32))

        rpn_features, roi_features = self.fcis.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.fcis.rpn(
            rpn_features, img_size, scales)
        rpn_locs = F.concat(rpn_locs, axis=0)
        rpn_scores = F.concat(rpn_scores, axis=0)

        gt_rpn_locs = []
        gt_rpn_labels = []
        for bbox in bboxes:
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                bbox, anchor, img_size)
            if cuda.get_array_module(rpn_locs.array) != np:
                gt_rpn_loc = cuda.to_gpu(gt_rpn_loc)
                gt_rpn_label = cuda.to_gpu(gt_rpn_label)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            del gt_rpn_loc, gt_rpn_label
        gt_rpn_locs = self.xp.concatenate(gt_rpn_locs, axis=0)
        gt_rpn_labels = self.xp.concatenate(gt_rpn_labels, axis=0)

        batch_indices = range(batch_size)
        sample_rois = []
        sample_roi_indices = []
        gt_roi_masks = []
        gt_roi_labels = []
        gt_roi_locs = []

        for batch_index, mask, label, bbox in \
                zip(batch_indices, masks, labels, bboxes):
            roi = rois[roi_indices == batch_index]
            sample_roi, gt_roi_mask, gt_roi_label, gt_roi_loc = \
                self.proposal_target_creator(
                    roi, mask, label, bbox, self.loc_normalize_mean,
                    self.loc_normalize_std, self.mask_size)
            del roi
            sample_roi_index = self.xp.full(
                (len(sample_roi),), batch_index, dtype=np.int32)
            sample_rois.append(sample_roi)
            sample_roi_indices.append(sample_roi_index)
            del sample_roi, sample_roi_index
            gt_roi_masks.append(gt_roi_mask)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_locs.append(gt_roi_loc)
            del gt_roi_mask, gt_roi_label, gt_roi_loc
        sample_rois = self.xp.concatenate(sample_rois, axis=0)
        sample_roi_indices = self.xp.concatenate(sample_roi_indices, axis=0)
        gt_roi_masks = self.xp.concatenate(gt_roi_masks, axis=0)
        gt_roi_labels = self.xp.concatenate(gt_roi_labels, axis=0)
        gt_roi_locs = self.xp.concatenate(gt_roi_locs, axis=0)

        roi_ag_seg_scores, roi_ag_locs, roi_cls_scores, _, _ = self.fcis.head(
            roi_features, sample_rois, sample_roi_indices,
            img_size, gt_roi_labels)

        # RPN losses
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs, gt_rpn_locs, gt_rpn_labels, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_scores, gt_rpn_labels)

        # Losses for outputs of the head
        n_roi = roi_ag_locs.shape[0]
        gt_roi_fg_labels = (gt_roi_labels > 0).astype(np.int)
        roi_locs = roi_ag_locs[self.xp.arange(n_roi), gt_roi_fg_labels]

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_locs, gt_roi_locs, gt_roi_labels, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_cls_scores, gt_roi_labels)
        # normalize by every (valid and invalid) instances
        roi_mask_loss = F.softmax_cross_entropy(
            roi_ag_seg_scores, gt_roi_masks, normalize=False) \
            * 10.0 / self.mask_size / self.mask_size

        loss = rpn_loc_loss + rpn_cls_loss \
            + roi_loc_loss + roi_cls_loss + roi_mask_loss
        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'roi_mask_loss': roi_mask_loss,
            'loss': loss,
        }, self)

        return loss
