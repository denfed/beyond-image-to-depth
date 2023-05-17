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

    with open("/data/sound-spaces/replica-visualechoes/scene_observations_128.pkl", 'rb') as f:
        obj1 = pickle.load(f)

    # print(obj)
    print(obj1.keys())
    print(obj1['apartment_0'].keys())
    # print(obj.keys(), len(obj))
    # print(obj['apartment_0'].keys())
    # print(obj['apartment_0'][(4, 0)])
    # display_sample(obj['apartment_0'][(4, 0)])

    # print("--------------DONE-----------------------")

    with open("/home/dcfedori/sound-spaces/data/scene_observations/replica/apartment_0.pkl", 'rb') as f:
        obj2 = pickle.load(f)


    # for i in obj1['apartment_0'].keys():
    #     print(obj1['apartment_0'][i])
    #     print(obj2[i])
    #     display_sample(obj1['apartment_0'][i])
    #     display_sample_semantic(obj2[i])

    # print(obj)
    # print(obj.keys())
    # print(obj[(4, 0)])
    # display_sample_semantic(obj[(4, 0)])

    # loop through scene observations to create big pickle

    dataset = {}

    directory = "/home/dcfedori/sound-spaces/data/scene_observations/replica/"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
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
            dataset[filename.replace(".pkl", "")] = obj

    # print(dataset, dataset.keys(), len(dataset.keys()))
    # print(obj1.keys(), len(obj1.keys()))

    # print(type(dataset), type(dataset['apartment_0']), type(dataset['apartment_0'][(4, 0)]), type(dataset['apartment_0'][(4, 0)]['rgb']))

    # print([type(k) for k in dataset.keys()])
    # print(dataset['apartment_0'][(4, 0)]['rgb'].shape,dataset['apartment_0'][(4, 0)]['depth'].shape,dataset['apartment_0'][(4, 0)]['semantic'].shape)
    with open('scene_observations_instance_semantic_128.pkl', 'wb') as psavefile:
        pickle.dump(dataset, psavefile)

    print("original depth", obj1['apartment_0'][(4, 0)]['depth'], obj1['apartment_0'][(4, 0)]['depth'].shape)
    print("new dpeth", dataset['apartment_0'][(4, 0)]['depth'], dataset['apartment_0'][(4, 0)]['depth'].shape)

