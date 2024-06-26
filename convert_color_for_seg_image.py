from email.mime import base
import json
import os 
import shutil
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def convert_tgt_style(im_src_path, im_tgt_path, beta=0.01, mode='RGB'):
    
    # RGB convert fda (3 channels)
    if mode == 'RGB':
        im_src = Image.open(im_src_path)
        im_tgt = Image.open(im_tgt_path).convert('RGB')
        im_tgt = im_tgt.resize(im_src.size, Image.BICUBIC )

        im_src_np = np.asarray(im_src, np.float32)
        im_tgt_np = np.asarray(im_tgt, np.float32)

        im_src_np = im_src_np.transpose((2, 0, 1))
        im_tgt_np = im_tgt_np.transpose((2, 0, 1))

        src_in_trg = FDA_source_to_target_np(im_src_np, im_tgt_np, L=beta)

        src_in_trg = src_in_trg.transpose((1,2,0))

        rgb_image = toimage(src_in_trg, cmin=0.0, cmax=255.0)
        fda_image = rgb_image
    # YUV sapce
    elif 'YCbCr' in mode:    
        # img_src = im_src.convert('YCbCr')
        img_src = Image.open(im_src_path).convert('YCbCr')
        img_tgt = Image.open(im_tgt_path).convert('YCbCr')
        img_tgt = img_tgt.resize(img_src.size, Image.BICUBIC )

        img_src = np.asarray(img_src, np.float32)
        img_tgt = np.asarray(img_tgt, np.float32)

        img_src = img_src.transpose((2, 0, 1))
        img_tgt = img_tgt.transpose((2, 0, 1))

        img_src_Y = np.reshape(img_src[0], (1,) + img_src[0].shape)
        img_tgt_Y = np.reshape(img_tgt[0], (1,) + img_tgt[0].shape )

        img_src_CbCr = img_src[1:]
        img_tgt_CbCr = img_tgt[1:]
        
        # applying fft in to CbCb channels
        if '-CbCr' in mode:
            src_in_tgt_CbCr = FDA_source_to_target_np(img_src_CbCr, img_tgt_CbCr, L=beta)
            
            src_in_tgt_YCbCr_2 = np.concatenate((img_src_Y, src_in_tgt_CbCr), axis=0)
            src_in_tgt_YCbCr_2 = src_in_tgt_YCbCr_2.transpose((1,2,0))

            yuv_2_image = toimage(src_in_tgt_YCbCr_2, cmin=0.0, cmax=255.0, mode='YCbCr')
            yuv_2_image_rgb = yuv_2_image.convert('RGB')

            fda_image = yuv_2_image_rgb
        else:
            # YUV convert by applying fft into Y channel
            src_in_tgt_Y = FDA_source_to_target_np(img_src_Y, img_tgt_Y, L=beta)

            src_in_tgt_YCbCr_1 = np.concatenate((src_in_tgt_Y, img_src_CbCr), axis=0)
            src_in_tgt_YCbCr_1 = src_in_tgt_YCbCr_1.transpose((1,2,0))

            yuv_1_image = toimage(src_in_tgt_YCbCr_1, cmin=0.0, cmax=255.0, mode='YCbCr')
            yuv_1_image_rgb = yuv_1_image.convert('RGB')

            fda_image = yuv_1_image_rgb

    return fda_image


with open('dir.json') as f:
    dirs = json.load(f)['dirs']

target_imgs_name = []

for dir_info in dirs:
    type = dir_info['type']
    position_label = dir_info.get('position_label', -1)
    damage_label = dir_info.get('damage_label', -1)
    seg_label = dir_info.get('segmentation_label', 0)
    location = dir_info['location']
    img_folder_name = dir_info.get('img_folder_name', '')
    img_file_extension = dir_info.get('img_file_extension', '')
    mask_folder_name = dir_info.get('mask_folder_name', '')
    mask_file_extension = dir_info.get('mask_file_extension', '')

    if type == 'segmentation' or position_label != 1:
        continue 

    for color_mode in ['FICE', 'WLI', 'BLI', 'LCI']:
        files_name = os.listdir(location + '/' + color_mode)
        files_name = [location + '/' + color_mode + '/' + fn for fn in files_name]
        target_imgs_name += files_name

beta_path_mapping = {
    0.01: '/home/kc/manhnd/new_data/ut_thuc_quan_1/images',
    0.02: '/home/kc/manhnd/new_data/ut_thuc_quan_2/images',
    0.05: '/home/kc/manhnd/new_data/ut_thuc_quan_5/images',
    0.09: '/home/kc/manhnd/new_data/ut_thuc_quan_9/images'
}

for beta in beta_path_mapping:
    base_path = beta_path_mapping[beta]

    if os.path.exists(base_path):
        for file_name in os.listdir(base_path):
            os.remove(base_path + '/' + file_name)

    src_base_path = '/home/kc/manhnd/DATA/UT thuc quan/20211021 UT thuc quan/images'
    src_imgs_name = os.listdir(src_base_path)

    for img_name in tqdm(src_imgs_name):
        target_img_name = random.choice(target_imgs_name)

        src_img_path = src_base_path + '/' + img_name
        tgt_img_path = target_img_name

        new_img = convert_tgt_style(src_img_path, tgt_img_path, beta=beta, mode='YCbCr-CbCr')
        new_img.save(base_path + '/' + img_name)
        # break

    print("%s: Done" % base_path)