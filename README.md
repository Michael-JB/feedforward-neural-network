# feedforward-neural-network

A simple implementation of a feedforward neural network with a sigmoid activation function and backpropagation training. While the network can be repurposed to suit simple supervised learning tasks, the default task calculates the bitwise AND, OR and XOR of a two-neuron input.

## Usage

- Install dependencies: `pip install -r requirements.txt`
- Run: `python neuralnet.py`

```
Training network...
Presenting input: [0, 0]
Expected output: [0, 0, 0]
Network output: [0.00022196 0.02425447 0.03501084]
Network output (rounded): [0. 0. 0.]

Presenting input: [0, 1]
Expected output: [0, 1, 1]
Network output: [0.02457758 0.98431149 0.96514629]
Network output (rounded): [0. 1. 1.]

Presenting input: [1, 0]
Expected output: [0, 1, 1]
Network output: [0.02231378 0.9843856  0.96538649]
Network output (rounded): [0. 1. 1.]

Presenting input: [1, 1]
Expected output: [1, 1, 0]
Network output: [0.96875556 0.99971883 0.03888555]
Network output (rounded): [1. 1. 0.]
```

## Built with

- [Python](https://www.python.org/) - Language
- [Numpy](https://numpy.org/) - Linear algebra library

## License

[MIT license](./LICENSE)