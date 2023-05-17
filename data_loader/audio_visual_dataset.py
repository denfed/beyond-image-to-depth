import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
from scipy.io import wavfile


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)
    spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    #print(spectro_two_channel.shape)
    return spectro_two_channel

def process_image(rgb, augment):
    if augment:
        # print('Doing Augmentation')
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(random.random()*0.6 + 0.7)
    return rgb


def parse_all_data(root_path, scenes):
    data_idx_all = []
    print(root_path)
    with open(root_path, 'rb') as f:
        data_dict = pickle.load(f)
    for scene in scenes:
        print(scene)
        data_idx_all += ['/'.join([scene, str(loc), str(ori)]) \
            for (loc,ori) in list(data_dict[scene].keys())]
        print(len(data_idx_all))    
    
    return data_idx_all, data_dict
        
class AudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        if self.opt.audio_normalize:
            audio = normalize(audio)

        #get the spectrogram of both channel
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)

        # # Get semantic segmentation map
        # semseg = self.data[scene][int(loc), int(orn)][('semantic')]
        # print(semseg, semseg.shape)

        # print("UNIQUE", np.unique(semseg))
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
        
        
        return {'img': img, 'depth':depth, 'audio':audio_spec_both}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'AudioVisualDataset'


class AudioVisualOnTheFlyDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

        self.base_audio_path = self.opt.audio_path

        # if self.opt.dataset == 'mp3d':
        #     self.audio_type = '3ms_sweep_16khz'
        # if self.opt.dataset == 'replica':
        #     self.audio_type = '3ms_sweep'

        self.rir_len = 44100

        self.rir_normalization = 10
        # possible options: 
        # "norm": l2 norm
        # <int>: amplitude scaling factor (divide rir by <int>)
        # None: none 

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        rir_path = os.path.join(self.base_audio_path, scene, orn, loc+'.wav')

        # audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        sr, rir = wavfile.read(rir_path)
        # Remove first 128 zero samples. This same operation is done in getEchoes.py
        rir = rir[128:]

        # Pad rir to 1s length. Validated that this does not affect final convolutions.
        rir_pad = np.zeros((self.rir_len, 2))
        rir_pad[:rir.shape[0], :] = rir

        # Normalize RIRs
        if isinstance(self.rir_normalization, int):
            rir_pad = rir_pad / self.rir_normalization
        elif self.rir_normalization == "norm":
            raise NotImplementedError
        elif self.rir_normalization is None:
            pass
        else:
            raise NotImplementedError

        #get the spectrogram of both channel
        # audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)

        # # Get semantic segmentation map
        # semseg = self.data[scene][int(loc), int(orn)][('semantic')]
        # print(semseg, semseg.shape)

        # print("UNIQUE", np.unique(semseg))
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
        
        # Change RIR to be channel first
        return {'img': img, 'depth':depth, 'rir':rir_pad.T}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'AudioVisualOnTheFlyDataset'


class SemanticAudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        if self.opt.audio_normalize:
            audio = normalize(audio)

        #get the spectrogram of both channel
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)

        # # Get semantic segmentation map
        semseg = torch.LongTensor(self.data[scene][int(loc), int(orn)][('semantic')])
        # semseg = semseg.unsqueeze(0)
        # print(semseg, semseg.shape)

        # print("UNIQUE", np.unique(semseg))
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
                semseg = semseg[:, left:left+128,upper:upper+128]


        # print(semseg.shape, img.shape, depth.shape)
        
        
        return {'img': img, 'depth':depth, 'semantic': semseg, 'audio':audio_spec_both}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'SemanticAudioVisualDataset'



class WaveformAudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        if self.opt.audio_normalize:
            audio = normalize(audio)

        #get the spectrogram of both channel
        # audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]
        
        
        return {'img': img, 'depth':depth, 'audio':audio}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'WaveformAudioVisualDataset'


class MultiviewAudioVisualDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        if self.opt.dataset == 'mp3d':
            self.data_idx, self.data = parse_all_data(
                                        os.path.join(self.opt.img_path, opt.mode+'.pkl'),
                                        self.opt.scenes[opt.mode])
            self.win_length = 32
        if self.opt.dataset == 'replica':
            self.data_idx, self.data = parse_all_data(self.opt.img_path, 
                                                        self.opt.scenes[opt.mode])   
            self.win_length = 64
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.base_audio_path = self.opt.audio_path
        if self.opt.dataset == 'mp3d':
            self.audio_type = '3ms_sweep_16khz'
        if self.opt.dataset == 'replica':
            self.audio_type = '3ms_sweep'

        self.orientations = [0, 90, 180, 270]

    def __getitem__(self, index):
        #load audio
        scene, loc, orn = self.data_idx[index].split('/')
        # print("DATASET INFO: ", scene, loc, orn)
        audio_path = os.path.join(self.base_audio_path, scene, self.audio_type, orn, loc+'.wav')
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate, mono=False, duration=self.opt.audio_length)
        if self.opt.audio_normalize:
            audio = normalize(audio)

        #get the spectrogram of both channel
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))
        #get the rgb image and depth image
        img = Image.fromarray(self.data[scene][(int(loc),int(orn))][('rgb')]).convert('RGB')
        
        if self.opt.mode == "train":
           img = process_image(img, self.opt.enable_img_augmentation)

        if self.opt.image_transform:
            img = self.vision_transform(img)

        depth = torch.FloatTensor(self.data[scene][(int(loc),int(orn))][('depth')])
        depth = depth.unsqueeze(0)

        # # Get semantic segmentation map
        # semseg = self.data[scene][int(loc), int(orn)][('semantic')]
        # print(semseg, semseg.shape)

        # print("UNIQUE", np.unique(semseg))
        
        if self.opt.mode == "train":
            if self.opt.enable_cropping:
                RESOLUTION = self.opt.image_resolution
                w_offset =  RESOLUTION - 128
                h_offset = RESOLUTION - 128
                left = random.randrange(0, w_offset + 1)
                upper = random.randrange(0, h_offset + 1)
                img = img[:, left:left+128,upper:upper+128]
                depth = depth[:, left:left+128,upper:upper+128]

        multiview = torch.zeros((4, 3, self.opt.image_resolution, self.opt.image_resolution))

        # Get images at every view
        for idx, i in enumerate(self.orientations):
            img_at_i = Image.fromarray(self.data[scene][(int(loc),int(i))][('rgb')]).convert('RGB')
            if self.opt.mode == "train":
                img_at_i = process_image(img_at_i, self.opt.enable_img_augmentation)

            img_at_i = self.vision_transform(img_at_i)

            if self.opt.mode == "train":
                if self.opt.enable_cropping:
                    RESOLUTION = self.opt.image_resolution
                    w_offset =  RESOLUTION - 128
                    h_offset = RESOLUTION - 128
                    left = random.randrange(0, w_offset + 1)
                    upper = random.randrange(0, h_offset + 1)
                    img_at_i = img_at_i[:, left:left+128,upper:upper+128]

            # print("multiview single image", img_at_i.shape)
            multiview[idx] = img_at_i
            # print("multiview", multiview.shape)
        # print(multiview)

        # add orientation metadata
        met = torch.eye(4)
        met = torch.roll(met, int(orn)//90, dims=1)
        # met = met.repeat(b, 1, 1)
        # print("metadata shape", met, met.shape, orn, int(orn)//90)
        
        
        return {'img': img, 'depth':depth, 'audio':audio_spec_both, 'multiview': multiview, 'orientation': int(orn)//90, 'or_metadata': met}

    def __len__(self):
        return len(self.data_idx)

    def name(self):
        return 'MultiviewAudioVisualDataset'