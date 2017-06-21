function evaluate(opt, net, dataset)
    net:evaluate()

    local batchSize = opt.batchSize
    local iterCnt, valLoss = 0, 0
    for startIdx = 1, dataset.size, opt.batchSize do
        local endIdx = math.min(startIdx + batchSize - 1, dataset.size)
        local size = (endIdx - startIdx + 1)

        local batchInput = torch.Tensor(size, 28, 28)
        for offset = 0, (size-1) do
            local image = dataset.data[startIdx + offset]
            batchInput[offset+1] = image
        end

        if opt.gpu == 1 then
            batchInput = batchInput:cuda()
            opt.criterion = opt.criterion:cuda()
        end

        local batchOutput = net:forward(batchInput)
        valLoss = valLoss + opt.criterion:forward(batchOutput, batchInput)
        iterCnt = (iterCnt + 1)
    end

    net:training()
    return valLoss / iterCnt 
end
