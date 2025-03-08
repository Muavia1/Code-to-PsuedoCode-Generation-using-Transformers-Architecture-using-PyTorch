# Code-to-PsuedoCode-Generation-using-Transformers-Architecture-using-PyTorch


**Code2PseudoCode** is a transformer-based deep learning model that translates programming code into human-readable pseudocode. This project aims to assist developers, students, and educators by providing a structured way to understand complex code through natural language.

## Features
- Utilizes a transformer architecture for sequence-to-sequence translation.
- Supports tokenization and vocabulary building for both code and pseudocode.
- Implements positional encoding and attention mechanisms to enhance translation accuracy.
- Uses PyTorch for efficient model training and inference.

## Installation
Clone the repository and install the dependencies:
```bash
git clone <repo-link>
cd Code2PseudoCode
pip install -r requirements.txt
```

## Usage
Run the model training script:
```bash
python train.py
```
To test the model, use:
```bash
python predict.py --input "your_code_here"
```

## Requirements
- Python 3.x
- PyTorch
- Pandas
- tqdm

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

---
