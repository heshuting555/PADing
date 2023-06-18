import torch
import argparse


def convert_weight(task_name, input_file):
    check = torch.load(input_file)
    model_weight = check['model']
    trainer_weight = check['trainer']
    iteration_weight = check['iteration']

    if task_name == 'panoptic':
        pass
    elif task_name == 'instance':
        if model_weight['sem_seg_head.predictor.class_embed.weight'].shape[0] == 66:
            num_class_train = 66
            num_class_all = 81
        else:
            num_class_train = 49
            num_class_all = 66
        for key in model_weight:
            if 'class_embed.weight' in key:
                classifier_weight_new = torch.zeros([num_class_all, 256], device='cuda')
                classifier_weight_new[:num_class_train] = model_weight[key]
                model_weight[key] = classifier_weight_new
            if 'class_embed.bias' in key:
                classifier_weight_new = torch.zeros([num_class_all], device='cuda')
                classifier_weight_new[:num_class_train] = model_weight[key]
                model_weight[key] = classifier_weight_new
    elif task_name == 'semantic':
        num_class_train = 157
        num_class_all = 172
        for key in model_weight:
            if 'class_embed.weight' in key:
                classifier_weight_new = torch.zeros([num_class_all, 256], device='cuda')
                classifier_weight_new[:num_class_train] = model_weight[key]
                model_weight[key] = classifier_weight_new
            if 'class_embed.bias' in key:
                classifier_weight_new = torch.zeros([num_class_all], device='cuda')
                classifier_weight_new[:num_class_train] = model_weight[key]
                model_weight[key] = classifier_weight_new
    else:
        print('please choose one task from panoptic, instance and semantic')

    model_weight['new_fc.weight'] = model_weight['sem_seg_head.predictor.class_embed.weight']
    model_weight['new_fc.bias'] = model_weight['sem_seg_head.predictor.class_embed.bias']
    checkpoint = {'model': model_weight, 'trainer': trainer_weight, 'iteration': iteration_weight}
    torch.save(checkpoint, 'pretrained_weight_{}.pth'.format(task_name))

    print('saving done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="panoptic")
    parser.add_argument("--input_file", type=str, default="supervise/model_final.pth")
    args = parser.parse_args()
    convert_weight(args.task_name, args.input_file)


if __name__ == "__main__":
    main()