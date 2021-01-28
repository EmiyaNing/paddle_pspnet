# This is a code for pspnet and unet's implement in paddle..

## Paddle debug first principle
1. see the information about your code, rather paddle's code....bug always come from your code, not paddle's code

### The first verison of this repositories 
1. finished the pspnet.py
2. finished the unet.py
3. add some print code to resnet_dilated.py

### To do list
1. finish the dataloader.py
2. finish the data reforience model
3. Train those two model in AIStudio

### version 2
1. finished the dataloader
2. add a seg_loss.py to compute the loss
3. add a utils.py
4. add a trains.py to train the model....

#### version 2 Bug list
1. in basic_data_load.py line 13 the label's resize function input should be label, but I write as data
2. In unet.py the model's output size is wrong.... 
3. In basic_data_load.py the opencv read a image's each pixel value as uint8 , but the type we need is int64...so how to change this type accuracy is a problem...
> Now we find the error is come from dataset. the dataset's image 2008_000009.png's pixel value have 59....
4. In trains.py Now the bug is train_loss is nan.....why it appear the bug "Not a number"????
> By print the seg_loss.py's variable preds, I find the model's predicts have a negative value....
> So I set the Unet's last layer's active function as relu....
5. Now the trains.py has a bug, the loss value is too big.........
6. Now I set the Unet's last layer's active function as softmax........
> By Searching in baidu, I find that if we use cross_entropy as loss function, we must use softmax active function in model's last layer..
> Because the cross_entropy function's input value must vary from 0 to 1...
