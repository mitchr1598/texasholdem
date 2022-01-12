from playingcards import Card, Deck, CardCollection
from dataclasses import dataclass
from collections import Counter


class PokerCollection(CardCollection):
    """
    A wrapper for CardCollection. Introduces properties important to poker
    """
    def __init__(self, card, maximum):
        super().__init__(card, maximum, ordered=True, reverse_order=True)

    @property
    def tone(self):
        return len(set([card.suit for card in self.cards]))

    @property
    def paired(self):
        return 2 in Counter([c.rank for c in self.cards]).values()


class TexasHand(PokerCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=2)


class Flop(PokerCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=3)


class Turn(PokerCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=1)


class River(PokerCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=1)


@dataclass
class Board:
    flop: Flop
    turn: Turn = None
    river: River = None

    def __post_init__(self):
        self.cards = PokerCollection(self.flop + self.turn + [self.river.cards[0]], maximum=5)

    @classmethod
    def from_string(cls, board_str):
        board_str = board_str.replace(' ', '')
        flop = Flop.from_string(board_str[:6])
        turn = Turn.from_string(board_str[6:8]) if len(board_str) > 6 else None
        river = River.from_string(board_str[8:]) if len(board_str) > 8 else None
        return Board(flop, turn, river)


def main():
    deck = Deck()
    deck.shuffle()
    hand = TexasHand.from_string('AdKc')
    print(type(hand))
    print(hand.ascii())


if __name__ == '__main__':
    main()
