import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from typing import Tuple, Optional
import re
import unicodedata
import argparse

@dataclass
class Config:
    """
    Configuration for data processing.
    """
    data_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.csv')
    has_header: bool = False
    test_size: float = 0.2
    random_state: int = 42
    out_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    lowercase: bool = False
    remove_special: bool = True
    unicode_normalize: bool = True
    logging_level: str = 'INFO'

def load_data(path: str, has_header: bool = False) -> pd.DataFrame:
    """
    Load the LinkedIn post dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    has_header : bool
        Whether the CSV file includes a header row.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.ParserError
        If the file cannot be parsed.
    """
    try:
        if has_header:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, header=None, names=["text", "category", "tone"])
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}")
        raise e
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV: {e}")
        raise e


def clean_text(text: str) -> str:
    """
    Basic text normalization for LinkedIn posts.

    Args:
        text (str): Raw post text.
    Returns:
        str: Cleaned text.
    """
    # Example cleaning: strip whitespace, replace double spaces, remove quotes
    cleaned = text.strip().replace('  ', ' ').replace('"', '')
    return cleaned


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the dataset.

    Args:
        df (pd.DataFrame): Raw dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df["text"] = df["text"].apply(clean_text)
    logging.info("Applied text cleaning.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and validation sets.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        test_size (float): Fraction for validation set.
        random_state (int): Seed for reproducibility.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logging.info(f"Split data: train={train_df.shape}, val={val_df.shape}")
    return train_df, val_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str = None) -> None:
    """
    Save train and validation splits to CSV files.

    Args:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        out_dir (str): Output directory (defaults to data/).
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    train_path = os.path.join(out_dir, 'train.csv')
    val_path = os.path.join(out_dir, 'val.csv')
    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    logging.info(f"Saved train split to {train_path}")
    logging.info(f"Saved val split to {val_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data processing.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="LinkedGen Data Processing")
    parser.add_argument('--data-path', type=str, default=Config.data_path, help='Path to input CSV file')
    parser.add_argument('--has-header', action='store_true', help='Specify if CSV has a header row')
    parser.add_argument('--test-size', type=float, default=Config.test_size, help='Validation set fraction')
    parser.add_argument('--random-state', type=int, default=Config.random_state, help='Random seed')
    parser.add_argument('--out-dir', type=str, default=Config.out_dir, help='Output directory for splits')
    parser.add_argument('--lowercase', action='store_true', help='Lowercase text during cleaning')
    parser.add_argument('--remove-special', action='store_true', help='Remove special characters during cleaning')
    parser.add_argument('--no-remove-special', action='store_true', help='Do not remove special characters')
    parser.add_argument('--unicode-normalize', action='store_true', help='Apply unicode normalization during cleaning')
    parser.add_argument('--no-unicode-normalize', action='store_true', help='Do not apply unicode normalization')
    parser.add_argument('--logging-level', type=str, default=Config.logging_level, help='Logging level (INFO, DEBUG, etc.)')
    return parser.parse_args()


def main(config: Config) -> None:
    """
    Main function to run data processing pipeline.
    """
    df = load_data(config.data_path)
    df = preprocess_data(df)
    train_df, val_df = split_data(df, config.test_size, config.random_state)
    save_splits(train_df, val_df, config.out_dir)
    logging.info("Data processing complete.")


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        data_path=args.data_path,
        has_header=args.has_header,
        test_size=args.test_size,
        random_state=args.random_state,
        out_dir=args.out_dir,
        lowercase=args.lowercase,
        remove_special=(not args.no_remove_special if args.remove_special or args.no_remove_special else Config.remove_special),
        unicode_normalize=(not args.no_unicode_normalize if args.unicode_normalize or args.no_unicode_normalize else Config.unicode_normalize),
        logging_level=args.logging_level
    )
    main(config)
