require 'optim'

function trainEpoch(opt, net, params, gradParams, dataset)
    local shuffle = torch.randperm( dataset['trainset'].size )
    local batchSize = opt.batchSize
    local currentLoss = 0

    ---------------------------------------------------------------------------
    -------------------- Create mini-batch for training -----------------------
    ---------------------------------------------------------------------------
    local iterCnt = 0
    for startIdx = 1, dataset['trainset'].size, batchSize do
        local endIdx = math.min(startIdx + batchSize - 1, dataset['trainset'].size)
        local size = (endIdx - startIdx + 1)
        
        local batchInput = torch.Tensor(size, 28, 28)
        for offset = 0, (size-1) do
            local image = dataset['trainset'].data[shuffle[startIdx + offset]]
            batchInput[offset+1] = image
        end

        if opt.gpu == 1 then
            batchInput = batchInput:cuda()
            opt.criterion = opt.criterion:cuda()
        end

    ---------------------------------------------------------------------------
    ---------------- Closure to evaluate loss and gradient  -------------------
    ---------------------------------------------------------------------------

        function feval(x)
            if x~= params then params:copy(x) end
            gradParams:zero()

            local batchOutput = net:forward(batchInput)
            local batchLoss = opt.criterion:forward(batchOutput, batchInput)
            local dloss_dout = opt.criterion:backward(batchOutput, batchInput)
            net:backward(batchInput, dloss_dout)

            return batchLoss, gradParams
        end

        _, fs = optim.sgd(feval, params, opt.sgd_params)
        currentLoss = currentLoss + fs[1]
        iterCnt = (iterCnt + 1)
    end
    return currentLoss / iterCnt
end
