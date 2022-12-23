import playingcards
from playingcards import Card, Deck, CardCollection
from dataclasses import dataclass
from collections import Counter
from typing import Union, Literal

TEXAS_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']


class PokerCollection(CardCollection):
    """
    A wrapper for CardCollection. Introduces properties important to poker
    """
    def __init__(self, cards, maximum, ordered=False):
        super().__init__(cards, maximum, ordered=ordered, reverse_order=True)
        self._quads = []
        self._trips = []
        self._pairs = []
        self._qtp_found = False  # Flag for lazy loading of quads, trips, and pairs

    @property
    def tone(self) -> int:
        return len(set([card.suit for card in self.cards]))

    @property
    def n_to_flush(self) -> int:
        """ How many in collection to a flush.
        Counts the number occurences of the most frequent suit """
        return Counter([card.suit for card in self.cards]).most_common(1)[0][1]

    @property
    def paired(self) -> bool:
        return 2 in Counter([c.rank for c in self.cards]).values()

    @property
    def quads(self) -> list[playingcards.Rank]:
        """ Returns a list of rankings for all instances of quads """
        if not self._qtp_found:
            self._qtp()
        return self._quads

    @property
    def trips(self) -> list[playingcards.Rank]:
        """ Returns a list of rankings for all instances of trips """
        if not self._qtp_found:
            self._qtp()
        return self._trips

    @property
    def pairs(self) -> list[playingcards.Rank]:
        """ Returns a list of rankings for all instances of pairs """
        if not self._qtp_found:
            self._qtp()
        return self._pairs

    def _qtp(self):
        """ Finds the quads, trips and pairs in one pass """
        for rank, count in self.ranking_counts.most_common():
            if count == 4:
                self._quads.append(rank)
            elif count == 3:
                self._trips.append(rank)
            elif count == 2:
                self._pairs.append(rank)
        self._qtp_found = True


class TexasHand(PokerCollection):
    def __init__(self, cards: list[Card], ordered=True):
        super().__init__(cards, maximum=2, ordered=ordered)


class Flop(PokerCollection):
    def __init__(self, cards: list[Card], ordered=True, **kwargs):
        super().__init__(cards, ordered=ordered, maximum=3)

    def _order_cards(self):
        if self.ordered:
            self.cards.sort(reverse=self.reverse_order)
            if self.paired and (self.cards[0].rank == self.cards[1].rank):  # Top card paired
                # Make sure the dominant suit is first
                if self.cards[1].suit == self.cards[2].suit:
                    self.cards[0], self.cards[1] = self.cards[1], self.cards[0]
            elif self.cards[1].rank == self.cards[2].rank and self.cards[0].suit == self.cards[2].suit:
                # Bottom card paired, and the end card is the dominant suit
                # Make sure the dominant suit is in the middle
                self.cards[1], self.cards[2] = self.cards[2], self.cards[1]


class Turn(PokerCollection):
    def __init__(self, cards: Union[Card, list[Card]], ordered=True):
        super().__init__(cards if isinstance(cards, list) else [cards], maximum=1, ordered=ordered)

    @property
    def rank(self):
        return self.rankings[0]

    @property
    def suit(self):
        return self.suits[0]


class River(PokerCollection):
    def __init__(self, cards: Union[Card, list[Card]], ordered=True):
        super().__init__(cards if isinstance(cards, list) else [cards], maximum=1, ordered=ordered)

    @property
    def rank(self):
        return self.rankings[0]

    @property
    def suit(self):
        return self.suits[0]


class Board(PokerCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=5)
        self.cards = Flop(self.cards[:3]).cards + self.cards[3:]  # Sorting the flop

    @property
    def flop(self):
        return Flop(self.cards[:3])

    @property
    def turn(self):
        return Turn(self.cards[3]) if len(self.cards) >= 4 else None

    @property
    def river(self):
        return River(self.cards[4]) if len(self.cards) >= 5 else None

    def generate_runnouts(self, street: Literal['turn', 'river'] = 'river', as_board=True) -> list[Union['Board', list]]:
        """
        Generates all possible runouts for the board
        :param street: The street to generate runouts until
        :param as_board: If True, returns a list of Board objects. If False, returns a list of lists of Cards
        """
        deck = Deck()
        deck.remove_cards(self.cards)
        if street == 'turn':
            runnouts = [[card] for card in deck]
        else:
            runnouts = [[card1, card2] for card1 in deck for card2 in deck if card1 != card2]
        if as_board:
            return [Board(self.cards + runnout) for runnout in runnouts]
        return runnouts

    def __eq__(self, other):
        return self.flop == other.flop and self.turn == other.turn and self.river == other.river

    # @property
    # def frontdoor_flush(self):
    #     return not self.flop.tone == 1 and

    @classmethod
    def from_ftr(cls, flop: Flop, turn: Turn = None, river: River = None):
        return cls.from_collections(flop, turn, river)


class TexasDeck(Deck):
    def __init__(self):
        cards = [
            Card(playingcards.Rank(r, num_rank), playingcards.Suit(s, pretty_suit))
            for s, pretty_suit in zip(playingcards.main.STANDARD_SUITS, playingcards.main.STANDARD_SUITS_PRETTY)
            for num_rank, r in enumerate(TEXAS_RANKS, start=2)
        ]
        super().__init__(cards)


def main():
    for c in sorted(TexasDeck().cards):
        print(c)


if __name__ == '__main__':
    main()
