from setuptools import setup, find_packages

setup(
    name="lara",
    version="0.1.0",
    description="LARA: Long-horizon Action-diffusion Robotic Agent",
    author="Eashan Vytla",
    author_email="Vytla.4@osu.edu",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "av",
        # Core dependencies
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        # Hydra for configuration
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        # LLM API
        "anthropic>=0.18.0",
        # Data handling
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        # PyTorch ecosystem
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "scikit-learn>=1.3.0",
        "datasets",
        "lerobot",
        "huggingface-hub",
        "pyarrow"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lara-generate-eval-gt=llm_evaluator.generate_eval_gt:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
