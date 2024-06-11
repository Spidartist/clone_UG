import onnxruntime
import numpy as np
import cv2

session = onnxruntime.InferenceSession("UG_UNET.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[3].name

# img_path = '/home/kc/manhnd/DATA/UT thuc quan/20211021 UT thuc quan/images/12.jpeg'
img_path = '/home/kc/manhnd/DATA/viem_loet_hanh_ta_trang_20220110/images/129.jpeg'
img = cv2.imread(img_path).astype(np.float32)
img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
img = np.transpose(img, (2, 0, 1))
img /= 255.
img = np.expand_dims(img, axis=0)

res1 = session.run([], {input_name: img})

# res1 = session.run([output_name], {input_name: img})[0]
# print(res1.shape)
# res1 = 1 / (1 + np.exp(-res1))

# res1[res1 < 0.5] = 0
# res1[res1 >= 0.5] = 1

# res1 = np.transpose(res1.reshape(1, 480, 480), (1, 2, 0)) * 255.
# cv2.imwrite('new_img.png', res1)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)