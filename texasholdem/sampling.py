import time
from collections import Counter
import itertools
from typing import Literal, Union
from time import perf_counter
import pickle
import os

import numpy as np
import pandas as pd
from playingcards import CardCollection
from tqdm import tqdm
from sqlalchemy import create_engine
from urllib.parse import quote_plus

import texasholdem
import playingcards

from texasholdem import texas_collections


ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


_SUIT_ORDER = (
    playingcards.Suit('s', '♠'),
    playingcards.Suit('d', '♦'),
    playingcards.Suit('c', '♣'),
    playingcards.Suit('h', '♥')
)


class BoardSample:
    def __init__(self, n: int, street: Literal['flop', 'turn', 'river']):
        """
        Sample a number of boards from the given street.
        :param n: The number of boards to sample.
        :param street: The street to sample.
        :return:
        """
        self.n = n
        self.street = street
        self._sample_df = self._generate_sample()

    @property
    def boards(self):
        return [texas_collections.Board.from_string(board) for board in self._sample_df.index]

    @property
    def to_dataframe(self):
        return self._sample_df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, street: Literal['flop', 'turn', 'river']):
        """
        Create a BoardSample from a dataframe. The dataframe must have a column 'count' and a column 'board'.
        :param df: The dataframe to create the BoardSample from.
        :param street: The street of the BoardSample.
        :return: A BoardSample.
        """
        if 'count' not in df.columns or 'board' not in df.columns:
            raise ValueError(f"df must have columns 'count' and 'board'. Got {df.columns}")
        df.set_index('board', inplace=True)
        df.index.name = 'board'
        obj = cls.__new__(cls)
        obj.n = df['count'].sum()
        obj.street = street
        obj._sample_df = df
        return obj

    def to_counter(self):
        key_cols = {'flop': 'flop', 'turn': ['flop', 'turn'], 'river': ['flop', 'turn', 'river']}[self.street]
        return counter_df_to_counter(self._sample_df, key_cols=key_cols, count_col='count')

    def search(self, sub_board: Union[texas_collections.Board, str]) -> list[texas_collections.Board]:
        """
        Search for all sampled boards which begin with sub_board.
        :param sub_board:
        :return:
        """
        if isinstance(sub_board, str):
            sub_board = texas_collections.Board.from_string(sub_board)
        if len(sub_board) > len(self.boards[0]):
            raise ValueError(f'Invalid sub_board: {sub_board}')
        return [b for b in self.boards if b.starts_with(sub_board)]

    def board_weight(self, board: Union[texas_collections.Board, str]):
        """
        Get's the sample weight of a given board.
        :param board:
        :return:
        """
        return self._sample_df['count'].loc[board]

    def runnouts(self, sub_board: Union[texas_collections.Board, str], turn_only=False) -> list[list[playingcards.Card]]:
        """
        Get's potential runouts from a board subset. Eg, if it's a river sample, and both 'AsKsJs9s8s' and
        'As Ks Js 6s 5s' are in the sample, then self.runnouts('As Ks Js') will return '9s8s' and '6s5s'.

        If turn_only is True, then it'll only give the turn card. Does not raise error if sample is a turn sample anyway.

        :param sub_board: The board subset to get runouts for.
        :param turn_only: If True, only return the turn card.
        :return: A list of runouts. Each runout is a list of cards of type playingcards.card
        """
        if isinstance(sub_board, str):
            sub_board = texas_collections.Board.from_string(sub_board)
        matches = self.search(sub_board)
        if turn_only:
            runnouts = []
            for b in matches:  # If a river sample, then there will be double ups of turns. So filtering
                if [b.turn] not in runnouts:
                    runnouts.append([b.turn])
            return runnouts
        else:
            return [b[len(sub_board):] for b in matches]

    def _generate_sample(self) -> pd.DataFrame:
        return sample_street(self.street, self.n, 'df')


class EmptyDataFrameError(Exception):
    pass


class NoPickleError(Exception):
    pass


def sample_street(street: Literal['flop', 'turn', 'river'],
                  n: int,
                  as_type: Literal['Counter', 'df'] = 'Counter'
                  ) -> Union[Counter, pd.DataFrame]:
    """
    Samples a given street from the database and returns it as a Counter or DataFrame.
    :param street: The street to sample from. Must be 'flop', 'turn', or 'river'
    :param n: The number of samples to take without replacement
    :param as_type: The type to return. Either 'Counter' or 'df'
    :return: Either a Counter or a DataFrame, depending on as_type. DataFrame will have columns 'count', 'flop'(, 'turn', 'river')
    """
    deck = texas_collections.TexasDeck()
    cards_in_sample = {'flop': 3, 'turn': 4, 'river': 5}[street]

    raw_samples = set()
    sample = []
    samples_done = 0
    while samples_done < n:
        deck.reset()
        deck.shuffle()
        flop = deck.draw_top_n(3)
        flop.ordered = True
        flop.reverse_order = True
        flop.order_cards()

        runnout = deck.draw_top_n(cards_in_sample - 3) if cards_in_sample > 3 else CardCollection()
        runnout_str = runnout.str_value(no_space=True) if cards_in_sample > 3 else ''
        raw_sample = flop.str_value(no_space=True) + runnout_str
        if raw_sample in raw_samples:
            continue
        raw_samples.add(raw_sample)

        board = flop + runnout
        suit_mapping = board.normalize_suit_mapping(_SUIT_ORDER)
        board = board.change_suits(suit_mapping)
        new_sample = board.str_value(no_space=True)
        sample.append(new_sample)
        samples_done += 1
    counter = Counter(sample)
    if as_type == 'df':
        return counter_to_df(counter, street)


def counter_to_df(counter: Counter, street: Literal['flop', 'turn', 'river']) -> pd.DataFrame:
    """
    Converts a Counter to a DataFrame. The Counter keys are the index, and the Counter values are the 'count' column.
    :param counter: The Counter to convert.
    :param street: The street of the Counter.
    :return: A DataFrame.
    """
    df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
    df.index.name = street
    return df


def counter_df_to_counter(df: pd.DataFrame, key_cols: Union[str, list, None] = None, count_col=None) -> Counter[str]:
    """
    Convert a dataframe to a counter. If key_cols is not specified, the index is used for the Counter keys.
    If count_col is not specified, there should only be one other column in the dataframe, which will be used as the
    counter values.
    :param df: The dataframe to convert. This dataframe should have a single column (or index) of values, and a single
    column of counts.
    :param key_cols: The column(s) to use as the keys for the Counter. If not specified, the index is used.
    :param count_col: The column to use as the counts for the Counter. If not specified, the only other column is used.
    :return: A Counter object.
    """
    if key_cols is not None:
        if isinstance(key_cols, list):
            df['key_col'] = df[key_cols].apply(lambda x: ''.join(x), axis='columns')  # Create new key column via concatenation
            df = df.drop(key_cols, axis='columns')  # Drop the old key columns
            key_cols = 'key_col'  # Set key_cols to the new key column
        # key_cols is now a single column name, and other key columns have been dropped
        df.set_index(key_cols, inplace=True)
    if count_col is None:
        if len(df.columns) != 1:
            raise ValueError(f'If count_col is not specified, df must have only one column. Got {len(df.columns)}')
        count_col = df.columns[0]
    return Counter(df.to_dict()[count_col])


if __name__ == '__main__':
    t0 = time.perf_counter()
    v = BoardSample(10800, 'river')
    t1 = time.perf_counter()
    print(f'Generated {v.n} boards ({len(v.boards)} unique) in {t1 - t0:.2f} seconds')
    print(v.to_dataframe.sort_values('count', ascending=False).head(10))
    print()

