import time
from collections import Counter
import itertools
from typing import Literal, Union
from time import perf_counter
import pickle
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from urllib.parse import quote_plus

import texasholdem
import playingcards

from texasholdem import texas_collections


ENGINE = create_engine(f'postgresql+psycopg2://postgres:{os.environ.get("PGDB_LOCAL_USER_PASSWORD")}@localhost/pokersolver')

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


_SUIT_ORDER = (
    playingcards.Suit('s', '♠'),
    playingcards.Suit('d', '♦'),
    playingcards.Suit('c', '♣'),
    playingcards.Suit('h', '♥')
)


class BoardSample:
    # TODO: Super slow as it generates every possible board for the given street
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
        df = sample_street(self.street, self.n, 'df')
        if self.street == 'flop':
            df['board'] = df['flop']
        elif self.street == 'turn':
            df['board'] = df['flop'] + '' + df['turn']
        elif self.street == 'river':
            df['board'] = df['flop'] + '' + df['turn'] + '' + df['river']
        df.set_index('board', inplace=True)
        df.index.name = 'board'
        return df[['count']]


class EmptyDataFrameError(Exception):
    pass

class NoPickleError(Exception):
    pass


def generate_every_flop() -> (Counter[str], list[texasholdem.Flop]):
    try:
        print('Attempting to load flops from database')
        df = load_counts_from_db('flop')
        if df.empty:
            raise EmptyDataFrameError("Connection succeeded, but returned empty dataframe")
        c = counter_df_to_counter(df, key_cols=['flop'], count_col='weight')
        returning = c, [texasholdem.Flop.from_string(f) for f in c.keys()]
        print('Successfully loaded flops from database')
    except EmptyDataFrameError:  # TODO: Also catch connection errors
        print('Failed to load flops from database')
        try:
            returning = load_street_from_pickle()
            print('Successfully loaded flops from pickle')
        except NoPickleError:
            print('Failed to load flops from pickle')
            returning = regenerate_every_flop()
            print('Successfully regenerated flops')

    return returning


def generate_every_turn() -> (Counter[str], list[texasholdem.Turn]):
    try:
        print('Attempting to load turns from database')
        df = load_counts_from_db('turn')
        if df.empty:
            raise EmptyDataFrameError("Connection succeeded, but returned empty dataframe")
        c = counter_df_to_counter(df, key_cols=['flop', 'turn'], count_col='weight')
        returning = c, [texasholdem.Board.from_string(t) for t in c.keys()]
        print('Successfully loaded turns from database')
    except EmptyDataFrameError:  # TODO: Also catch connection errors
        print('Failed to load turns from database')
        try:
            returning = load_street_from_pickle('turn')
            print('Successfully loaded turns from pickle')
        except NoPickleError:
            print('Failed to load turns from pickle')
            returning = regenerate_every_turn()
            print('Successfully regenerated turns')

    return returning


def generate_every_river() -> (Counter[str], list[texasholdem.River]):
    try:
        print('Attempting to load rivers from database')
        df = load_counts_from_db('river')
        if df.empty:
            raise EmptyDataFrameError("Connection succeeded, but returned empty dataframe")
        c = counter_df_to_counter(df, key_cols=['flop', 'turn', 'river'], count_col='weight')
        returning = c, [texasholdem.Board.from_string(r) for r in c.keys()]
        print('Successfully loaded rivers from database')
    except EmptyDataFrameError:  # TODO: Also catch connection errors
        print('Failed to load rivers from database')
        try:
            returning = load_street_from_pickle('river')
            print('Successfully loaded rivers from pickle')
        except NoPickleError:
            print('Failed to load rivers from pickle')
            returning = regenerate_every_river()
            print('Successfully regenerated rivers')

    return returning


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
    sampling_fn = f'sample_{street}s' if n < 100000 else f'sample_{street}s_large'
    print(f'Attempting to sample {n} {street}s from database')
    df = pd.read_sql(f'select * from {sampling_fn} ({n})', ENGINE)
    print(f'Successfully sampled {n} {street}s from database')
    if as_type == 'df':
        return df
    key_cols = {'flop': 'flop', 'turn': ['flop', 'turn'], 'river': ['flop', 'turn', 'river']}[street]
    return counter_df_to_counter(df, key_cols=key_cols, count_col='count')


def load_counts_from_db(street: Literal['flop', 'turn', 'river']) -> pd.DataFrame:
    return pd.read_sql_table(f'strategic_{street}s_weights', ENGINE, schema='poker')


def load_street_from_pickle(street: Literal['flop', 'turn', 'river']) -> (Counter[str], list[texasholdem.Flop]):
    path = ABSOLUTE_PATH + f"/caching/generate_every_{street}.bin"
    if os.path.exists(path):
        with open(path, "rb") as f:  # "rb" because we want to read in binary mode
            loaded = pickle.load(f)
        return loaded
    else:
        raise NoPickleError(f'There is not pickle at "{path}"')


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


def regenerate_every_flop():
    deck = texasholdem.TexasDeck()
    gen = itertools.combinations(deck, 3)
    flop = [texasholdem.Flop(list(f)) for f in gen]
    norm_flop = [f.normalize_suits(_SUIT_ORDER).str_value() for f in flop]
    c = Counter(norm_flop)
    return c, [texasholdem.Flop.from_string(f) for f in c.keys()]


def regenerate_every_turn():
    deck = texasholdem.TexasDeck()
    flop_counter, all_flops = generate_every_flop()
    boards = []
    for flop in all_flops:
        deck.reset()
        deck.remove_cards(flop)
        for card in deck:
            if card in flop:
                continue
            if flop.tone == 1 and card.suit != flop.cards[0].suit:
                card = playingcards.Card(card.rank, playingcards.Suit('d', '♦'))
            elif flop.tone == 2 and card.suit not in flop.suits:
                card = playingcards.Card(card.rank, playingcards.Suit('c', '♣'))
            for _ in range(flop_counter[flop.str_value()]):
                boards.append(f'{flop.str_value()} {card.str_value()}')
    c = Counter(boards)
    returning = c, list([texasholdem.Board.from_string(k) for k in c.keys()])
    return returning


def regenerate_every_river():
    deck = texasholdem.TexasDeck()
    turn_counter, all_turns = generate_every_turn()
    boards = []
    for turn in tqdm(all_turns):
        deck.reset()
        deck.remove_cards(turn)
        for card in deck:
            if card in turn:
                continue
            if turn.tone == 1 and card.suit != turn.cards[0].suit:
                card = playingcards.Card(card.rank, playingcards.Suit('d', '♦'))
            elif turn.tone == 2 and card.suit not in turn.suits:
                card = playingcards.Card(card.rank, playingcards.Suit('c', '♣'))
            elif turn.tone == 3 and card.suit not in turn.suits:
                card = playingcards.Card(card.rank, playingcards.Suit('h', '♥'))
            for _ in range(turn_counter[turn.str_value()]):
                boards.append(f'{turn.str_value()} {card.str_value()}')
    c = Counter(boards)
    returning = c, list([texasholdem.Board.from_string(k) for k in c.keys()])
    return returning


if __name__ == '__main__':
    t0 = time.perf_counter()
    v = BoardSample(10000, 'flop')
    t1 = time.perf_counter()
    print(f'Generated {sum(v.values())} boards ({len(v)} unique) in {t1 - t0:.2f} seconds')
    print()

