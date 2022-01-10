from playingcards import Card, Deck, CardCollection


class TexasHand(CardCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=2)
        self.cards.sort(reverse=True)


class Flop(CardCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=3)
        self.cards.sort(reverse=True)


class Turn(CardCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=1)
        self.cards.sort(reverse=True)


class River(CardCollection):
    def __init__(self, cards: list[Card]):
        super().__init__(cards, maximum=1)
        self.cards.sort(reverse=True)


def main():
    deck = Deck()
    deck.shuffle()
    hand = TexasHand.from_string('AdKc')
    print(type(hand))
    print(hand.ascii())


if __name__ == '__main__':
    main()
