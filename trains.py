import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from unet  import UNet
from pspnet import PSPNet
from basic_data_load import BasicDataLoader,Transform
from seg_loss import Basic_SegLoss
#from basic_data_preprocessing import TrainAugmentation


parser = argparse.ArgumentParser()
#parser.add_argument('--net', type=str, default='basic')
parser.add_argument('--net', type=str, default='pspnet')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)


args = parser.parse_args()

def train(dataloader, model, criterion, optimizer, epoch, total_batch, psp_flag):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        image = data[0].astype("float32")
        label = data[1]
        image = fluid.layers.transpose(image, perm=(0, 3, 1, 2))
        if psp_flag:
            pred  = model(image)[0]
        else:
            pred  = model(image)
        loss  = criterion(pred, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg



def main():
    # Step 0: preparation
    place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        # Step 1: Define training dataloader
        transform  = Transform(256)
        dataloader = BasicDataLoader(image_folder=args.image_folder,
                                     image_list_file=args.image_list_file,
                                     transform=transform,
                                     shuffle=True)
        train_dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        train_dataloader.set_sample_generator(dataloader, batch_size = args.batch_size, places=place)
        total_batch      = int(len(dataloader) / args.batch_size)
        psp_flag         = False
        # Step 2: Create model
        if args.net == "basic":
            model = UNet(59)
        else:
            model = PSPNet(59)
            psp_flag = True

        # Step 3: Define criterion and optimizer
        criterion = Basic_SegLoss
        optimizer = AdamOptimizer(learning_rate=args.lr, parameter_list=model.parameters())
        
        # Step 4: Training
        for epoch in range(1, args.num_epochs+1):

            train_loss = train(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               epoch,
                               total_batch,
                               psp_flag)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss}")

            '''if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')'''



if __name__ == "__main__":
    main()
