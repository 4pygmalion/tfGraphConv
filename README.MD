#### Graph Convolution Layer
- implemented with TensorFlow 2.x

#### Requriment 
- tensorflow 2.x
- networkx
- numpy 


#### Usage
```python3
from GraphConvNet import GraphConvolution
from tensroflow as tf

model = tf.keras.model.Sequntial()

gc = GraphConvolution(input_dim=10, output_dim=30)
model.add(gc)
model.compile(...)

```

#### Install
``` bash
$ git clone https://github.com/4pygmalion/tfGraphConv.git
```