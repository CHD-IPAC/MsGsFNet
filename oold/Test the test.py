from test import test_model
import os

# patch_based = True
patch_based = False
# save_the_pred_img = True
save_the_pred_img = False

# net_reparam = False
net_reparam = True

if patch_based:
    rate_patchbased = 0.1
else:
    rate_patchbased = None
# model = 'CLMGNet'
model = 'MsGsFNet'
# model = 'segnetm'
# model = 'unetm'
# model = 'FreeNet'
# model = 'SS3FCN_rad'
# model = 'SS3FCN'
# model = 'FullyContNet'
# model = 'HPDM_SPRN'
# model = 'SSTN'
# model = 'GSCVIT'
# model = 'DBCTNet'
# model = 'repvit'


# model = 'S3ANet'
# model = 'small_ablation_only_spatial'
# model = 'lsk_ablation_only_spatial'
# model = 'MsGsFNet_baseline'
# model = 'SegFormer'
# model = 'small_addsmallkernel'
# model = 'small_lsk'

dataset = 'HongHu'
# dataset = 'LongKou'
# dataset = 'HanChuan'
# dataset = 'Qingyun'
# dataset = 'rad'

# dataset = 'Tangdaowan'
# dataset = 'Pingan'

# rate_trainset = '45'
# rate_trainset = '40'
# rate_trainset = '35'
# rate_trainset = '30'
rate_trainset = '25'
# rate_trainset = '20'
# rate_trainset = '15'
# rate_trainset = '10'
# rate_trainset = '5'
# rate_trainset = '4'
# rate_trainset = '3'
# rate_trainset = '2'
# rate_trainset = '1'
# rate_trainset = '0.5'

# patchsize = '4'
# patchsize = '8'
# patchsize = '16'
patchsize = '32'

if dataset == 'HongHu':
    bands = 270
    num_class = 23
    inner_nc = 64
elif dataset == 'LongKou':
    bands = 270
    num_class = 10
    inner_nc = 64
elif dataset == 'HanChuan':
    bands = 274
    num_class = 17
    inner_nc = 64
elif dataset == 'rad':
    bands = 51
    num_class = 6
    inner_nc = 256
elif dataset == 'Tangdaowan':
    bands = 176
    num_class = 19
    inner_nc = 256
elif dataset == 'Qingyun':
    bands = 176
    num_class = 7
    inner_nc = 64
elif dataset == 'Pingan':
    bands = 176
    num_class = 11
    inner_nc = 256


# 四个权重文件路径
# weight_files = {
#     # 'best_oa': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestoa_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
#     'best_miou': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestMiou_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
#     # 'best_miou': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights_sp//_bestMiou_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
#     # 'best_aa': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestaa_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
#     # 'best_mds': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestmds_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset)
# }

weight_files = {
    # 'best_oa': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestoa_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
    'best_miou': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestMiou_//{}//{} of patch{} in rate of {}.pt'.format(dataset, model, patchsize, rate_trainset),
    # 'best_miou': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights_sp//_bestMiou_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
    # 'best_aa': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestaa_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset),
    # 'best_mds': '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//_bestmds_//{}//{} in rate of {}.pt'.format(dataset, model, rate_trainset)
}

results = []

for key, weight_path in weight_files.items():
    if key == 'best_miou' and save_the_pred_img:
        save_pred = True
    else:
        save_pred = False
    print(f'\033[1;33mTesting {key} with weight file: {weight_path}\033[0m')
    Overall_accuracy, Average_Accuracy, Mean_IOU, Mean_DICE_score, FWIOU = test_model(weight_path, model, dataset, bands, num_class, rate_trainset, save_pred, patch_based, rate_patchbased, inner_nc, net_reparam, patchsize)
    result = {
        'file_name': weight_path,
        'test_case': key,
        'Overall_accuracy': float(Overall_accuracy),
        'Average_Accuracy': float(Average_Accuracy),
        'Mean_IOU': float(Mean_IOU),
        'Mean_DICE_score': float(Mean_DICE_score),
        'FWIOU': float(FWIOU)
    }
    results.append(result)


# # 先清空结果txt
if not save_the_pred_img:
    output_file = '//media/ubuntu/2ad0bba9-bef3-4ad8-b5ea-812c019ea962//zwq//oold//pretrainedweights//rad//The results of {} in {} with rate of {}'.format(model, dataset, rate_trainset)
    # 打开文件并清空内容
    with open(output_file, 'w') as f:
        pass  # pass 表示什么也不做，即清空文件
    print(f"文件 '{output_file}' 已成功清空。")
    # 将结果保存到文件
    with open(output_file, 'w') as f:
        for result in results:
            f.write(f"Test Case: {result['test_case']}\n")
            f.write(f"File: {result['file_name']}\n")
            f.write(f"Overall_accuracy: {result['Overall_accuracy']}\n")
            f.write(f"Average_Accuracy: {result['Average_Accuracy']}\n")
            f.write(f"Mean_IOU: {result['Mean_IOU']}\n")
            f.write(f"Mean_DICE_score: {result['Mean_DICE_score']}\n")
            f.write(f"FWIOU: {result['FWIOU']}\n\n")

    print(f"Results saved to {output_file}")
