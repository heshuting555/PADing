from argparse import ArgumentParser

from PADing.evaluation.ins_seg_evaluation_gzero_utils.coco_utils import coco_eval_revise as coco_eval


def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    parser.add_argument(
        '--classwise', action='store_true', help='whether eval class wise ap')
    parser.add_argument(
        '--gzsi', action='store_true', help='whether eval class with gzsd setting')
    parser.add_argument(
        '--num-seen-classes',
        type=int,
        default=48)
    args = parser.parse_args()
    coco_eval(args.result, args.types, args.ann, args.max_dets, args.classwise, args.gzsi, args.num_seen_classes)


if __name__ == '__main__':
    main()
