# Deep Learning Training Pipeline for CIFAR Image Classification

A modular and extensible deep learning training pipeline built with **PyTorch**, designed for large-scale experimentation on CIFAR-style image classification tasks.

This project demonstrates **end-to-end machine learning engineering skills**, including dataset management, data augmentation, model abstraction, training orchestration, and evaluation.

---

## Key Highlights

- **End-to-End Training Pipeline**  
  Implemented a complete workflow covering dataset downloading, preprocessing, training, and evaluation.

- **Modular Architecture Design**  
  Decoupled data augmentation, model definition, training logic, and evaluation metrics to enable rapid experimentation.

- **Configurable Experiments**  
  Supported flexible hyperparameter control via command-line arguments for batch size, learning rate, epochs, and datasets.

- **Reproducible and Scalable**  
  Clean project structure with explicit separation between source code and generated artifacts.

---

## Project Structure

.
├── main.py                     # Experiment entry point
├── scripts/
│   ├── __init__.py
│   ├── data_augmentation.py    # Data augmentation strategies
│   ├── data_download.py        # Dataset downloading and preprocessing
│   ├── model_architectures.py  # Neural network architectures
│   ├── train_utils.py          # Training and validation logic
│   └── evaluation_metrics.py   # Evaluation metrics

---

## Technical Stack

- Language: Python  
- Deep Learning Framework: PyTorch, torchvision  
- Data Processing: NumPy, Albumentations  
- Training Utilities: tqdm  
- Evaluation: scikit-learn  

---

## Running an Experiment

python main.py \
  --dataset cifar100 \
  --num_epochs 300 \
  --batch_size 128 \
  --lr 0.1 \
  --weight_decay 5e-4

Hyperparameters are configurable via command-line arguments, enabling fast iteration across different experimental settings.

---

## Engineering Details

- Automatic dataset downloading and preprocessing  
- Encapsulated data augmentation strategies  
- Centralized model architecture definitions  
- Clean separation between training logic and evaluation metrics  

---

## Why This Project Matters

This repository reflects practical skills in:
- Designing clean, maintainable ML codebases
- Building reproducible experiment pipelines
- Translating research-style experiments into production-ready structure
- Writing scalable training logic rather than single-use scripts

It serves as a strong foundation for further work in computer vision, reinforcement learning, and applied machine learning systems.

---

## Notes

- Large artifacts such as datasets and model checkpoints are intentionally excluded from version control.
- This project focuses on engineering structure and experimentation rather than distributing pre-trained models.

---

## License

For academic and educational use.
