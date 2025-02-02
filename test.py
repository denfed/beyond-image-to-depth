
import os 
import torch
import numpy as np
from options.test_options import TestOptions
import torchvision.transforms as transforms
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import compute_errors, compute_region_errors
from models import criterion 
import wandb

loss_criterion = criterion.LogDepthLoss()
opt = TestOptions().parse()
opt.device = torch.device("cuda")

run = wandb.init(project="material-echo",
				 entity="dcfedori",
				 job_type="test",
				 name=opt.name,
                 mode="disabled" if opt.nowandb else "online"
				 )

run.log_code(".")

builder = ModelBuilder()
net_audiodepth = builder.build_audiodepth(opt.audio_shape,
                    weights=os.path.join(opt.checkpoints_dir, 'audiodepth_'+opt.dataset+'.pth'))
net_rgbdepth = builder.build_rgbdepth(
                    weights=os.path.join(opt.checkpoints_dir, 'rgbdepth_'+opt.dataset+'.pth'))
net_attention = builder.build_attention(
                    weights=os.path.join(opt.checkpoints_dir, 'attention_'+opt.dataset+'.pth'))
net_material = builder.build_material_property(
                    weights=os.path.join(opt.checkpoints_dir, 'material_'+opt.dataset+'.pth'))
nets = (net_rgbdepth, net_audiodepth, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)
model.eval()

wandb.watch(model)

opt.mode = 'test'
dataloader_val = CustomDatasetDataLoader(opt)
dataloader_val.initialize(opt)
dataset_val = dataloader_val.load_data()
dataset_size_val = len(dataloader_val)
print('#validation clips = %d' % dataset_size_val)




losses, errs = [], []
errs_img = []
errs_audio = []
with torch.no_grad():
    for i, val_data in enumerate(dataset_val):
        output = model.forward(val_data)
        depth_predicted = output['depth_predicted']
        depth_gt = output['depth_gt']
        # print(depth_gt.shape)
        # print(depth_predicted, depth_gt)
        # input()
        img_depth = output['img_depth']
        audio_depth = output['audio_depth']
        attention = output['attention']
        loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
        losses.append(loss.item())
        
        for idx in range(depth_gt.shape[0]):
            errs.append(compute_errors(depth_gt[idx].cpu().numpy(), 
                                depth_predicted[idx].cpu().numpy()))

            errs_img.append(compute_region_errors(depth_gt[idx].cpu().numpy(), 
                                img_depth[idx].cpu().numpy(), minr=0.4*opt.max_depth, maxr=1.0*opt.max_depth))

            errs_audio.append(compute_region_errors(depth_gt[idx].cpu().numpy(), 
                                audio_depth[idx].cpu().numpy(), minr=0.4*opt.max_depth, maxr=1.0*opt.max_depth))
            
                 
mean_loss = sum(losses)/len(losses)
mean_errs = np.array(errs).mean(0)

print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs[1])) 

errors = {}
errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs[0], mean_errs[1], mean_errs[5]
errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs[2], mean_errs[3], mean_errs[4]
errors['MAE'] = mean_errs[6]

wandb.log({"RMSE": errors['RMSE'],
			"ABS_REL": errors['ABS_REL'],
			"DELTA-1": errors['DELTA1'],
			"DELTA-2": errors['DELTA2'],
			"DELTA-3": errors['DELTA3'],
            "MAE": errors['MAE']})

print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
print('==='*25)


mean_errs_img = np.array(errs_img).mean(0)

print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs_img[1])) 

errors = {}
errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs_img[0], mean_errs_img[1], mean_errs_img[5]
errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs_img[2], mean_errs_img[3], mean_errs_img[4]
errors['MAE'] = mean_errs[6]

print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
print('==='*25)

mean_errs_audio = np.array(errs_audio).mean(0)

print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errs_audio[1])) 

errors = {}
errors['ABS_REL'], errors['RMSE'], errors['LOG10'] = mean_errs_audio[0], mean_errs_audio[1], mean_errs_audio[5]
errors['DELTA1'], errors['DELTA2'], errors['DELTA3'] = mean_errs_audio[2], mean_errs_audio[3], mean_errs_audio[4]
errors['MAE'] = mean_errs_audio[6]

wandb.log({"RMSE": errors['RMSE'],
			"ABS_REL": errors['ABS_REL'],
			"DELTA-1": errors['DELTA1'],
			"DELTA-2": errors['DELTA2'],
			"DELTA-3": errors['DELTA3'],
            "MAE": errors['MAE']})

print('ABS_REL:{:.3f}, LOG10:{:.3f}, MAE:{:.3f}'.format(errors['ABS_REL'], errors['LOG10'], errors['MAE']))
print('DELTA1:{:.3f}, DELTA2:{:.3f}, DELTA3:{:.3f}'.format(errors['DELTA1'], errors['DELTA2'], errors['DELTA3']))
print('==='*25)