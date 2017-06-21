mnist = require 'mnist'

function getDataset()
    local fullset = mnist.traindataset()
    
    local dataset = {}
    dataset.trainset = {
        size = 50000,
        data = fullset.data[ {{1,50000}} ]:double(),
        label = fullset.label[ {{1, 50000} }]
    }
    
    dataset.validationset = {
        size = 10000,
        data = fullset.data[ {{50001, 60000}} ]:double(),
        label = fullset.label[ {{50001, 60000}} ]
    }
    
    dataset.testset = mnist.testdataset()
    dataset.testset.data = dataset.testset.data:double()
    return dataset
end
