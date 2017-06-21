require 'model'
require 'dataset'
require 'trainAE'
require 'evaluateAE'
require 'image'

---------------------------------------------------------------------------
-------------------- Parse command-line parameters ------------------------
---------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text("Training model for face recognition")
cmd:text()
cmd:text("Options")
cmd:option("-gpu", 0, "set this flag to 1 if you want to use GPU")
cmd:option("-layerSize", 49, "the number of nodes in the bottleneck layer")
cmd:option("-batchSize", 200, "the batch size for training")
cmd:option("-numEpochs", 100, "the number of epochs to train")

opt = cmd:parse(arg)
if opt.gpu == 1 then
    cunnOk, cunn = pcall(require, "cunn")
    cutorchOk, cutorch = pcall(require, "cutorch")
    
    if not cunnOk or not cutorchOk then
        print ("cunn and/or cutorch are not properly configured.")
        print ("Falling back to CPU mode...")
        opt.gpu = 0
    end
end

---------------------------------------------------------------------------
------------------------ Loading net and dataset --------------------------
---------------------------------------------------------------------------
ae = getAE(opt)
params, gradParams = ae:getParameters()
dataset = getDataset()

---------------------------------------------------------------------------
------------------------ Train model on dataset ---------------------------
---------------------------------------------------------------------------

opt.criterion = nn.MSECriterion()
opt.sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 1e-3,
    momentum = 1e-4
}

opt.itersInEpoch = math.ceil(dataset['trainset'].size / opt.batchSize)
for epochCtr = 1, opt.numEpochs do
    local trainLoss = trainEpoch(opt, ae, params, gradParams, dataset)
    local valLoss = evaluate(opt, ae, dataset['validationset'])
    print ("epoch " .. epochCtr .. "/" .. opt.numEpochs .. ": trainLoss = " .. trainLoss .. ", valLoss = " .. valLoss)
end

testLoss = evaluate(opt, ae, dataset['testset'])
print ("testLoss = " .. testLoss)

---------------------------------------------------------------------------
------------------------ Visualize features ---------------------------
---------------------------------------------------------------------------
linear = ae.modules[4]
basis = torch.eye(opt.layerSize)

translate = nn.Sequential()
translate:add(linear)
translate:add(nn.Reshape(28, 28))

image.display(translate:forward(basis))
