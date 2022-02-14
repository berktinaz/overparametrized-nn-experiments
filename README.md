# overparametrized-nn-experiments

* Experiment 1: Effect of overparametrization
    * Final settings: 
        * inputDim = 3
        * hiddenSize = 3
        * overFactor = 1, 100
        * dataSize = 48
        * LR = 0.33333
        * epochs = 2000
        * seed = 0
* Experiment 2: Effect of initialization scale
    * Final settings: 
        * inputDim = 3
        * hiddenSize = 3
        * overFactor = 1000
        * dataSize = 48
        * LR = 0.01
        * target = 1e-8
        * seed = 100
        * scales = 0.01,0.03,0.1,0.32,1,3,10,14
* Experiment 3: Visualizing weights (non-Lazy vs Lazy regime)
    * Final settings: 
        * inputDim = 3
        * hiddenSize = 3
        * overFactor = 33
        * dataSize = 48
        * LR = 0.25
        * target = 1e-8
        * seed = 100
        * scales = 0.01,3
* Experiment 4: Visualizing learned function vs 2D training samples
    * Final settings: 
        * inputDim = 2
        * hiddenSize = 3
        * overFactor = 33
        * dataSize = 48
        * LR = 0.25
        * target = 1e-8
        * seed = 100
        * scales = 0.01,3