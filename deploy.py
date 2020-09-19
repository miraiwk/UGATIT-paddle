import os
import numpy as np
import matplotlib.pyplot as plt

from paddle import fluid

from dataset import ImageFolder
from utils.dataloader import DataLoader
from utils import transforms

def deploy(path):
    assert os.path.exists(path), f'{path} not found : ('
    dataset = 'YOUR_DATASET_NAME'

    img_size = 256
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    testA = ImageFolder(os.path.join('dataset', dataset, 'testA'), test_transform)
    with fluid.dygraph.guard(): 
        testA_loader = DataLoader(testA, batch_size=1, shuffle=False)
        real_A, _ = next(iter(testA_loader))
        in_np = real_A.numpy()

    # load model
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_vars, fetch_vars = fluid.io.load_inference_model(path, exe)

    # inference
    fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
    def img_postprocess(img):
        assert isinstance(img, np.ndarray), type(img)
        img = img * 0.5 + 0.5
        img = img.squeeze(0).transpose((1, 2, 0))
        # BGR to RGB
        img = img[:, :, ::-1]
        return img
    in_img = img_postprocess(in_np)
    out_img = img_postprocess(fetch)
    plt.subplot(121)
    plt.title('real A')
    plt.imshow(in_img)
    plt.subplot(122)
    plt.title('A to B')
    plt.imshow(out_img)
    plt.show()

if __name__ == '__main__':
    deploy_path = 'save_infer_model'
    deploy(deploy_path)
