require 'nn'

torch.manualSeed(0)

function getAE(opt)
    local model = nn.Sequential()
    model:add(nn.Reshape(28*28))
    model:add(nn.Linear(28*28, opt.layerSize))
    model:add(nn.Tanh())
    model:add(nn.Linear(opt.layerSize, 28*28))
    model:add(nn.Reshape(28, 28))
    
    if opt.gpu == 1 then model = model:cuda() end
    return model
end
