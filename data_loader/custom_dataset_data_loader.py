#!/usr/bin/env python

import torch.utils.data

def CreateDataset(opt):
    dataset = None
    from data_loader.audio_visual_dataset import AudioVisualDataset, WaveformAudioVisualDataset, SemanticAudioVisualDataset, MultiviewAudioVisualDataset

    if opt.waveformaudio or opt.audio_only_waveform:
        dataset = WaveformAudioVisualDataset()
        print("HERE")
    elif opt.semanticpyramid:
        dataset = SemanticAudioVisualDataset()
        print("Using semantic segmentation dataset.")
    elif opt.multiview:
        dataset = MultiviewAudioVisualDataset()
        print("Using multiview dataset.")
    else:
        dataset = AudioVisualDataset()
        # dataset = MultiviewAudioVisualDataset()

    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.dataset = CreateDataset(opt)
        shuff = False
        if opt.mode == "train":
            print('Shuffling the dataset....')
            shuff= True
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=shuff,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)//self.opt.batchSize

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
