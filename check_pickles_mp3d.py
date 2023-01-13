import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def display_sample_semantic(sample):
    img = sample["rgb"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

def display_sample(sample):
    img = sample["rgb"]
    depth = sample["depth"]

    arr = [img, depth]
    titles = ["rgba", "depth"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show(block=False)


if __name__ == "__main__":
    """
    A random script to just look at the pickle formats of the echolocation image data
    and see what is coming out of the soundspaces cache_observations.py.
    """

    with open("/home/cendue/dcfedori/datasets/mp3d-visualechoes/mp3d_split_wise/test.pkl", 'rb') as f:
        obj1 = pickle.load(f)

    with open('dataset/metadata/mp3d/mp3d_scenes_train.txt') as f:
        trainscenes = f.read().splitlines()

    with open('dataset/metadata/mp3d/mp3d_scenes_val.txt') as f:
        valscenes = f.read().splitlines()

    with open('dataset/metadata/mp3d/mp3d_scenes_test.txt') as f:
        testscenes = f.read().splitlines()

    print(trainscenes,valscenes,testscenes)

    # # loop through scene observations to create big pickle

    train_dataset = {}
    val_dataset = {}
    test_dataset = {}

    directory = "/home/cendue/dcfedori/sound-spaces/data/scene_observations/mp3d/"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        scene = filename.replace(".pkl", "")
        # # checking if it is a file
        # if os.path.isfile(f):
        print(f, filename)
        with open(f, "rb") as pfile:
            obj = pickle.load(pfile)
            for k, v in obj.items():
                # v['depth'] = np.squeeze(v['depth'], axis=2)
                # v['semantic'] = np.squeeze(v['semantic'], axis=2)
                obj[k] = dict(v)
                # print(v, v['depth'].shape)
            # print(obj.keys())
            if scene in trainscenes:
                train_dataset[scene] = obj
            elif scene in valscenes:
                val_dataset[scene] = obj
            elif scene in testscenes:
                test_dataset[scene] = obj
            else:
                print("scene", scene, "discarded")

    with open('mp3d_instance_semantic_train_128.pkl', 'wb') as psavefile:
        pickle.dump(train_dataset, psavefile)

    with open('mp3d_instance_semantic_val_128.pkl', 'wb') as psavefile:
        pickle.dump(val_dataset, psavefile)

    with open('mp3d_instance_semantic_test_128.pkl', 'wb') as psavefile:
        pickle.dump(test_dataset, psavefile)

    with open("/home/cendue/dcfedori/datasets/mp3d-visualechoes/mp3d_split_wise/test.pkl", 'rb') as f:
        obj1 = pickle.load(f)

    # display_sample(obj1['D7N2EKCX4Sj'][(0, 0)])

    with open('mp3d_instance_semantic_test_128.pkl', 'rb') as f:
        obj2 = pickle.load(f)

    print(obj1['XcA2TqTSSAj'].keys())
    print(obj1['XcA2TqTSSAj'][(1, 0)]['depth'])
    print(obj2['XcA2TqTSSAj'][(1, 0)]['depth'])
    print(obj1['XcA2TqTSSAj'][(1, 0)]['rgb'])
    print(obj2['XcA2TqTSSAj'][(1, 0)]['rgb'])

    # display_sample_semantic(obj1['D7N2EKCX4Sj'][(0, 0)])

