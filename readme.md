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