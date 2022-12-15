import re, os, random
import pandas as pd 

def preprocess_text(text: str) -> str:
    """
    1. Lowercase all letters
    2. Add a blank space around the following punctuations .,!?
    3. Replace all other puncturations with a whitespace

    Parameters
    ----------
    text : str
        String to be processed

    Returns
    -------
        processed string

    Examples
    --------
        Original String: "Hello, World! What's up?"
        Returned String: "hello , world ! what s up ? "
    """
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text) 
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text) 
    return text

def preprocess_yelp_csv_file(filepath: str, append_with: str) -> str:
    """
    Takes in a file path, preproesses each line of text using the preprocess_text
    and saves the file back. Assumes that the file can be loaded into memory

    Parameters
    ----------
    filepath: str
        Path to the file, which must be a csv file with a .csv extention
        of yelp data reviews in the format "rating", "review_text"
    
    append_with: str
        Creates a new file with the string passed to append_with, appended to the main file
        with an underscore in front
    
    Returns
    -------
    str
        The filepath for the processed file

    Example
    -------
        preprocess_yelp_csv_file("/Users/temp/data/yelp/train.csv", "processed")
        saves the file train_processed.csv to /Users/temp/data/yelp/
    """
    df = pd.read_csv(filepath_or_buffer=filepath, header=None, names=["label", "review"])
    df.review = df.review.apply(preprocess_text)
    new_filepath = filepath.strip(".csv") + f"_{append_with}.csv"
    df.to_csv(new_filepath)
    return new_filepath

def count_lines(filepath: str) -> int:
    """
    Count the number of lines in a file (similar to wc -l command in *nix)

    Parameters
    ----------
    filepath: str
        Path to the file
    
    Returns
    -------
    int
        The line count

    """
    count = 0
    with open(filepath, "r") as f:
        for line in f:
            count += 1
    return count

def split_yelp_data(filepath: str, split_ratios: list, split_filenames: list, shuffle: bool = True) -> None:
    """
    Takes the yelp csv file as input, and splits it into a training and validation split

    Parameters
    ----------
    filepath: str
        Path to the file
    
    split_ratios: list
        A list of two float values, the sum of the values must be 1.0
    
    split_filenames: list
        A list of strings with the filenames to save the splits into

    shuffle: bool
        If True, shuffles the file before splitting

    Returns
    -------
    None
        Saves the split files as two new files

    Example
    -------
        split_yelp_data(
            filepath = "/Users/temp/data/train.csv", 
            split_ratios = [0.6, 0.4], 
            split_filenames: ["train.csv", "validation.csv"], 
            shuffle = True
        )
    """
    assert len(split_ratios) == 2
    assert len(split_filenames) == 2
    assert round(sum(split_ratios),0) == 1.0
    num_lines = count_lines(filepath)
    line_nums = list(range(num_lines))
    if shuffle:
        random.shuffle(line_nums)
    file_1_line_nums = set(line_nums[:int(num_lines*split_ratios[0])])
    # file_2_line_nums = set(line_nums[int(num_lines*split_ratios[0]):])
    file_1_path = os.path.join(os.path.dirname(filepath), split_filenames[0])
    file_2_path = os.path.join(os.path.dirname(filepath), split_filenames[1])

    with open(filepath, "r") as fp, open(file_1_path, "w") as fp_1, open(file_2_path, "w") as fp_2:
        for i, line in enumerate(fp):
            if i in file_1_line_nums:
                fp_1.writelines(line)
            else:
                fp_2.writelines(line)
    
    return file_1_path, file_2_path
