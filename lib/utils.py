import cv2
import ref
import numpy as np
from plyfile import PlyData


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        return pkl.load(f)

def Crop_by_Pad(img, center, scale, res, channel=3, interpolation=cv2.INTER_NEAREST, resize=True):
    ht, wd = img.shape[0], img.shape[1]
    upper = max(0, int(center[0] - scale / 2. + 0.5))
    left  = max(0, int(center[1] - scale / 2. + 0.5))
    bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
    right  = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))
    crop_ht = float(bottom - upper)
    crop_wd = float(right - left)
    if crop_ht > crop_wd:
        resize_ht = res
        resize_wd = int(res / crop_ht * crop_wd + 0.5)
    elif crop_ht < crop_wd:
        resize_wd = res
        resize_ht = int(res / crop_wd * crop_ht + 0.5)
    else:
        resize_wd = resize_ht = res

    if channel == 3 or channel == 1:
        tmpImg = img[upper:bottom, left:right, :]
        if not resize:
            outImg = np.zeros((int(scale), int(scale), channel))
            outImg[int(scale / 2.0 - (bottom-upper) / 2.0 + 0.5):(int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom-upper)),
            int(scale / 2.0 - (right-left) / 2.0 + 0.5):(int(scale / 2.0 - (right-left) / 2.0 + 0.5) + (right-left)), :] = tmpImg
            return outImg
        try:
            resizeImg = cv2.resize(tmpImg, (resize_wd, resize_ht), interpolation=interpolation)
        except:
            # raise Exception
            return np.zeros((res, res, channel))

        if len(resizeImg.shape) < 3:
            resizeImg = np.expand_dims(resizeImg, axis=2) # for depth image, add the third dimension
        outImg = np.zeros((res, res, channel))
        outImg[int(res / 2.0 - resize_ht / 2.0 + 0.5):(int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht),
        int(res / 2.0 - resize_wd / 2.0 + 0.5):(int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd), :] = resizeImg
    else:
        raise NotImplementedError

    return outImg
    

def get_ply_model(path):
    ply = PlyData.read(path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)
