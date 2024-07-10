import os
from argparse import ArgumentParser
from collections import OrderedDict
from functools import cmp_to_key
from itertools import permutations
from typing import Dict, List, Optional, OrderedDict, Union


class BioVocabGenerator:
    def __init__(
        self,
        gram_num: Union[int, None] = None,
        sort: bool = True,
        cmp_list: Union[List[str], None] = None,
        aa_list: List[str] = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "O",
            "U",
            "B",
            "J",
            "Z",
            "X",
        ],
        # mmseqs2 aa list: (A S T) (C) (D B N) (E Q Z) (F Y) (G) (H) (I V) (K R) (L J M) (P) (W) (X)
        special_tokens: List[str] = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"],
    ) -> None:
        # 1. Set the gram_num for tokenization.
        # Example: gram_num = 3, 'ABCDE' -> ['ABC', 'BCD', 'CDE']
        if gram_num is not None:
            assert gram_num % 2 != 0, "gram_num must be odd!"
        self.gram_num = gram_num

        # 2. Set the amino acid list and add special_tokens for tokenization.
        self.aa_list = aa_list
        self.special_tokens = special_tokens

        # 3. Set the bool value for sort, cmp_dict is the dict order to sort.
        self.sort = sort
        self.cmp_dict = self.__fill_cmp_list(
            self.aa_list if cmp_list is None else cmp_list
        )

        if gram_num is not None:
            self.vocab = self.__generate_vocab
            self.vocab_dict = self.__generate_vocab_dict

    def __fill_cmp_list(self, cmp_list: List[str]) -> Dict[str, int]:
        """
        fill the start and end syntax for cmp_dict
        """

        return {value: index for index, value in enumerate(cmp_list + [">", "<"])}

    @property
    def __iter_list(self) -> List[str]:
        """
        generate iter_list for permutations
        ['A', 'B', 'C'] -> ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']
        """

        return [i for _ in range(self.gram_num) for i in self.aa_list] + [">", "<"]

    def __remove_errstr(self, x: str) -> bool:
        """
        remove error string from raw_vocab
        error str example: 'A>B', '<QW'
        """

        if x.count("<") + x.count(">") == 0:
            return True
        elif x.count("<") + x.count(">") == 1:
            if x[0] == ">" or x[-1] == "<":
                return True
        else:
            return False

    def __vocab_cmp(self, x: str, y: str) -> int:
        """
        cmp function for sort
        """

        for i, j in zip(x, y):
            if self.cmp_dict[i] < self.cmp_dict[j]:
                return -1
            elif self.cmp_dict[i] > self.cmp_dict[j]:
                return 1
            else:
                continue

    @property
    def __generate_vocab(self) -> List[str]:
        """
        generate n-mer amino acid vocabulary
        """
        # generate raw_vocab from permutations
        raw_vocab = permutations(self.__iter_list, r=self.gram_num)

        # use set to clear duplicate values and remove the error strs
        vocab = list(
            set(["".join(i) for i in raw_vocab if self.__remove_errstr(i) == True])
        )

        # sort the vocab
        if self.sort is True:
            vocab = sorted(vocab, key=cmp_to_key(self.__vocab_cmp))

        return self.special_tokens + vocab

    @property
    def __generate_vocab_dict(self) -> OrderedDict:
        """
        convert vocabulary from List to OrderedDict
        """

        return OrderedDict(zip(self.vocab, [i for i in range(len(self.vocab))]))

    def get_size(self) -> int:
        return len(self.vocab)

    def get_vocab_list(self) -> List[str]:
        return self.vocab

    def get_vocab_dict(self) -> OrderedDict:
        return self.vocab_dict

    def encode(self, input: str) -> int:
        try:
            token_id = int(self.vocab_dict[input])
        except KeyError as e:
            print("Can not find {} in vocabulary!".format(e))
        finally:
            return token_id

    def decode(self, index: int) -> str:
        return self.vocab[index]

    def save_vocabdict(self, path: Optional[str] = None) -> None:
        path_name = "vocab.txt"

        if path is None:
            path = path_name
        elif os.path.isdir(path):
            path += "/" + path_name

        try:
            with open(path, "w") as f:
                data = self.vocab_dict
                for i, j in data.items():
                    f.write("{0:>6} {1:>5}\n".format(i, str(j)))
        except:
            print("Writing Error!")


class BioVocabLoader(BioVocabGenerator):
    def __init__(self, path: str) -> None:
        super().__init__()
        assert os.path.exists(path), "vocab path not exists!"
        self.load_vocab_dict(path)
        self.get_gram_num()

    def load_vocab_dict(self, path: str) -> None:
        """
        load the vocabulary dictionary from txt
        """

        with open(path, "r") as f:
            data = [line.strip() for line in f.read().splitlines()]
            self.vocab = [i.split()[0] for i in data]
            self.vocab_dict = OrderedDict({i.split()[0]: i.split()[1] for i in data})

    def get_gram_num(self) -> None:
        """
        get the n-gram split from the vocabulary
        """

        if isinstance(self.gram_num, int):
            return self.gram_num
        else:
            for i in self.vocab:
                if i not in self.special_tokens:  # default 5 special_tokens
                    return len(i)


class BioTokenizer(BioVocabLoader):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Tokenizer hyperparameter.")
        parser.add_argument("--vocab_path", default=None, type=str)
        return parent_parser

    def __init__(self, args=None, vocab_path: str = 'AbLingua/tokens.txt') -> None:
        if vocab_path != None:
            super().__init__(vocab_path)
        elif args != None:
            super().__init__(args.vocab_path)
        

        self.gram_num = self.get_gram_num()

    def __cut_seq(self, seq: str) -> List[str]:
        """
        cut a sequence to 3-gram/3-mer token list
        ">ABCDE<" -> '>AB', 'ABC', 'BCD', 'CDE', 'DE<'
        """

        seq = seq.upper()
        assert len(seq) - self.gram_num + 1 > 0, "Protein sequence is too short to cut!"
        return [seq[i : i + self.gram_num] for i in range(len(seq) - self.gram_num + 1)]

    def __single_seq_tokenize(self, seq: str) -> List[int]:
        """
        convert token to index
        """

        # assert len(seq) > 10, 'Too short to process!'
        token_list = self.__cut_seq(seq)
        token_ids = [self.encode(i) for i in token_list]

        return token_ids

    def __append_headtail(self, seq: str) -> str:
        """
        append '>' on sequence head and '<' on sequence tail
        """

        if seq[0] != ">":
            seq = ">" + seq
        if seq[-1] != "<":
            seq += "<"

        return seq

    def get_token_list(self, seq: str) -> List[str]:
        """
        split sequence to a list contains all tokens
        """

        seq = self.__append_headtail(seq)

        assert len(seq) > 10, "Too short to process!"
        token_list = self.__cut_seq(seq)

        return token_list

    def tokenize(self, seq: str, pt: bool = False) -> List[int]:
        """
        tokenize the sequence to ids
        """

        assert seq.isalpha(), f"ERROR Seq: {seq}\nProtein Sequence has illegal char!"

        seq = self.__append_headtail(seq)
        token_ids = self.__single_seq_tokenize(seq)

        return token_ids

    def detokenize(self, ids: List[str]) -> str:
        """
        detokenize ids to sequence
        """

        seq = [self.decode(i) for i in ids]

        return seq