from collections import Counter
import itertools
from typing import Literal, Union

import numpy as np
import pandas as pd

import texasholdem
import playingcards

from texasholdem import texas_collections

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

    def search(self, sub_board: Union[texas_collections.Board, str]):
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
        return self._sample_df['n'].loc[board]

    def runnouts(self, sub_board: Union[texas_collections.Board, str]) -> list[list[playingcards.Card]]:
        """
        Get's potential runouts from a board subset. Eg, if it's a river sample, and both 'As Ks Js 9s 8s' and
        'As Ks Js 6s 5s' are in the sample, then self.runnouts('As Ks Js') will return '9s 8s' and '6s 5s'.
        :param sub_board: The board subset to get runouts for.
        :return: A list of runouts. Each runout is a list of cards of type playingcards.card
        """
        if isinstance(sub_board, str):
            sub_board = texas_collections.Board.from_string(sub_board)
        matches = self.search(sub_board)
        return [b[len(sub_board):] for b in matches]

    def _generate_sample(self):
        street_fn_map = {'flop': generate_every_flop, 'turn': generate_every_turn, 'river': generate_every_river}
        generate_board_fn = street_fn_map[self.street]
        counter, _ = generate_board_fn()
        flops, counts = zip(*counter.items())
        rng = np.random.default_rng()
        c = Counter(rng.choice(flops, size=self.n, replace=True, p=np.array(counts) / sum(counts)))
        df = pd.DataFrame(c.items(), columns=['board', 'n'])
        df.set_index('board', inplace=True)
        df.index.name = 'board'
        return df


def generate_every_flop() -> (Counter[str], list[texasholdem.Flop]):
    deck = texasholdem.TexasDeck()
    gen = itertools.combinations(deck, 3)
    flop = [texasholdem.Flop(list(f)) for f in gen]
    norm_flop = [f.normalize_suits(_SUIT_ORDER).str_value() for f in flop]
    c = Counter(norm_flop)
    return c, [texasholdem.Flop.from_string(f) for f in c.keys()]


def generate_every_turn():
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
    return c, list([texasholdem.Board.from_string(k) for k in c.keys()])


def generate_every_river():
    deck = texasholdem.TexasDeck()
    turn_counter, all_turns = generate_every_turn()
    boards = []
    for turn in all_turns:
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
    return c, list([texasholdem.Board.from_string(k) for k in c.keys()])


print(generate_every_turn() == generate_every_turn_faster())

