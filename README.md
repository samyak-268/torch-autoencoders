# torch-autoencoders
Implementation of a simple auto-encoder which is trained on the MNIST dataset

Pre-requisites:
+ MNIST Data Loader -- To install, run ``luarocks install mnist``
+ Torch7 Image Toolbox -- To install, run ``luarocks install image``

To start training, run ``th main.lua -gpu 1 -layerSize 50 -batchSize 200 -numEpochs 25``

Feel free to change the value of the parameters. Set the ``gpu`` param to 1,
only if ``cutorch`` and ``cunn`` are installed. Otherwise, use the default value of 0.
