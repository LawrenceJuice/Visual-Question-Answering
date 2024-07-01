import re
import torch
import pandas
import numpy as np
from PIL import Image
from statistics import mode
from transformers import BertTokenizer


def process_text(text):
    """
    Preprocesses the given text by performing various transformations.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text after applying the transformations.
    """

    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class VQADataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the VQA (Visual Question Answering) task.

    Args:
        df_path (str): The path to the JSON file containing image paths, questions, and answers.
        image_dir (str): The directory where the image files are located.
        transform (callable, optional): A function/transform to apply to the image. Default is None.
        answer (bool, optional): Whether to include answers in the dataset. Default is True.

    Attributes:
        transform (callable): The function/transform applied to the image.
        image_dir (str): The directory where the image files are located.
        df (pandas.DataFrame): The DataFrame containing image paths, questions, and answers.
        answer (bool): Whether answers are included in the dataset.
        question2idx (dict): A dictionary mapping question words to their corresponding indices.
        answer2idx (dict): A dictionary mapping answer words to their corresponding indices.
        idx2question (dict): A dictionary mapping question indices to their corresponding words.
        idx2answer (dict): A dictionary mapping answer indices to their corresponding words.

    Methods:
        update_dict(dataset): Updates the dictionaries of the dataset with the dictionaries from another dataset.
        __getitem__(idx): Retrieves the item at the given index.
        __len__(): Returns the length of the dataset.

    """

    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  
        self.image_dir = image_dir  
        self.df = pandas.read_json(df_path)
        self.answer = answer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            words = question.split(" ")
            for word in words:
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)


    def update_dict(self, dataset):
        """
        Updates the dictionaries of the dataset with the dictionaries from another dataset.

        Parameters:
            dataset (Dataset): The dataset from which to update the dictionaries.

        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer


    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, question, and answers (if available) at the given index.
                - image (PIL.Image.Image): The image at the given index.
                - question (torch.Tensor): The one-hot encoded question at the given index.
                - answers (torch.Tensor): The one-hot encoded answers at the given index (if available).
                - mode_answer_idx (int): The index of the mode answer (most frequent answer) at the given index.

        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        # Tokenize the question
        question_tokenized = self.tokenizer(self.df["question"][idx], return_tensors="pt", padding="max_length", max_length=512)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # Get mode (correct label)

            return image, question_tokenized, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question_tokenized
    def __len__(self):
        return len(self.df)
