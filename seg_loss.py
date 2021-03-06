import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def Basic_SegLoss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape
    n1, c1, h1, w1 = labels.shape
    # TODO: create softmax_with_cross_entropy criterion
    criterion = fluid.layers.cross_entropy
    # TODO: transpose preds to NxHxWxC
    preds = fluid.layers.transpose(preds, perm=(0, 2, 3, 1))
    # create a mask for preds, to prevent the pixel value over the 255
    mask = (labels!=ignore_index)
    mask = fluid.layers.cast(mask, 'float32')
    # TODO: call criterion and compute loss
    loss = criterion(preds, labels)
    if fluid.layers.has_nan(loss):
        print("Error, there has a nan...")
        print(preds)
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)
    return avg_loss

def main():
    label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000002.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64)
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:,:,np.newaxis]
    label = label[np.newaxis, :, :, :]

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = Basic_SegLoss(pred, label)
        print(loss)

if __name__ == "__main__":
    main()

