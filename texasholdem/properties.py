from texasholdem import texas_collections
from dataclasses import dataclass
from collections import Counter
import playingcards


#
@dataclass
class Property:  # Not using
    is_true: bool
    abs_strength_1: int
    abs_strength_2: int
    rel_strength_1: int
    rel_strength_2: int

    def __bool__(self):
        return self.is_true


@dataclass
class TexasHandProperties:
    # Constuct the logic first, then I think I need to look into the get_att, set_att methods. So much of this can be
    # cached. So much can be streamline. When one property is calculated, it can use stuff from previous properties.
    # Perhaps an post_initialization with them all as None, followed by using get_att and set_att to deal with the rest
    """ Determines what properties belong to a player """
    hand: texas_collections.TexasHand
    board: texas_collections.Board

    @property
    def all_cards(self) -> texas_collections.PokerCollection:
        return texas_collections.PokerCollection(self.hand + self.board, 7)

    @property
    def _flush_suit(self) -> playingcards.Suit:
        """ Determines if the player has a flush, greater than that of the board """
        # Will return true even if the flush is on the board. But the player still has a flush if that's the case
        suit_counts = self.all_cards.suit_counts  # Setting as new variable for multiple use, as caching isn't done yet
        best_suit, best_suit_count = suit_counts.most_common(1)[0]  # Most common returns a list of tuples
        if best_suit_count >= 5:
            return best_suit
        else:
            return playingcards.Suit('')

    @property
    def _straight_value(self) -> playingcards.Rank:
        """ Rank of value of player straight, ignoring flushes atm. Empty rank if no straight """
        # Could probably set the ordered descending tuple here. Useful in other areas such as straight-flush
        return _straight_value(self.all_cards.rankings)

    @property
    def _hit_count(self) -> Counter[playingcards.Rank, int]:
        """ Counts the number of times each card in hand hits the board """
        return Counter([rank for rank in self.board.ranking_counts if rank in self.hand.rankings])

    @property
    def straight_flush(self) -> playingcards.Card:
        """ Returns the best card that forms the straight flush. Empty card otherwise """
        if not self._flush_suit:  # Checking flush exists first
            return playingcards.Card(playingcards.Rank(''), playingcards.Suit(''))

        # Start with an empty card collection and add each flush card to it
        flush_cards = playingcards.CardCollection()
        for card in self.all_cards:
            if card.suit == self._flush_suit:
                flush_cards.add_cards([card])
        straight_rank = _straight_value(flush_cards.rankings)

        if not straight_rank:  # If straight doesn't exist then return empty card
            return playingcards.Card(playingcards.Rank(''), playingcards.Suit(''))

        # Now we know flush exists as there's a straight within those flush cards
        return playingcards.Card(straight_rank, self._flush_suit)  # un-cached flush value being called twice here

    @property
    def quads(self) -> list:
        """ Does the player have quads"""
        # Straight-flush not possible if quads exist, so overalls everything
        return self.all_cards.quads

    @property
    def full_house(self) -> tuple[playingcards.Rank, playingcards.Rank]:
        """ Does the player have quads a full house. Will not evaluate to false as it returns a tuple """
        if any([self.straight_flush, self.quads]):  # Check for stronger hands first
            return playingcards.Rank(''), playingcards.Rank('')

        _hc = self._hit_count  # Hit count # TODO: Not necessary once I have good caching
        _hc_mc = _hc.most_common()  # Hit Count Most Common -- list of tuples in desc order of occurance # TODO: Not necessary once I have good caching
        if (self.hand.paired and _hc[self.hand.rankings[0]] == 1) and len(self.board.pairs) > 0:  # Set on a paired board
            return self.hand.rankings[0], max(self.board.pairs)
        elif _hc_mc[0][1] == 2 and _hc_mc[1][1] == 1:  # Trips and pair kind of boat
            # Multiple strength cases
            #   - Double paired
            #       - Players second card plays (1)
            #       - Players second card doesn't play (2)
            #   - Single paired (3)
            if len(self.board.pairs) > 1:  # Double paired
                non_hit_pair = [rank for rank in self.board.pairs if rank not in self.hand.rankings][0]
                if _hc_mc[1][0] > non_hit_pair:  # "the pair" in the full-house is in play (1)
                    return _hc_mc[0][0], _hc_mc[1][0]
                else:  # "the pair" in the full-house is NOT in play (2)
                    return _hc_mc[0][0], non_hit_pair
            else:  # One pair on board (3)
                return _hc_mc[0][0], _hc_mc[1][0]
        elif 2 in self._hit_count and len(self.board.pairs) > 1:  # Trips on double paired board
            non_hit_pair = [rank for rank in self.board.pairs if rank not in self.hand.rankings][0]
            return _hc_mc[0][0], non_hit_pair
        else:
            return playingcards.Rank(''), playingcards.Rank('')

    @property
    def flush(self) -> playingcards.Card:
        """
        Does the player have a flush.
            - Returns the biggest flush card the player holds.
            - If playing flush on board then it returns an empty ranked card with the correct suit
            - If doesn't have a flush, then returns an empty card
        """
        if any([self.straight_flush, self.quads, self.full_house]):  # Check for stronger hands first
            return playingcards.Card(playingcards.Rank(''), playingcards.Suit(''))

        if self._flush_suit:
            if self.hand.filter_suit(self._flush_suit):
                return max(self.hand.filter_suit(self._flush_suit))
            else:
                return playingcards.Card(playingcards.Rank(''), self._flush_suit)
        else:
            return playingcards.Card(playingcards.Rank(''), playingcards.Suit(''))

    @property
    def straight(self):
        """ Returns the top rank of the straight """
        return self._straight_value


def _straight_value(ranks: list[playingcards.Rank]):  # I have test cases for these in "Python Plays/poker_property_testing"
    """ Returns the straight value of the ranks given, returns 0 if no straight """
    # It does this by "preparing" the ranks given, and then using dynamic programming
    # It turns the list input into a tuple, sorted in decending order. Then a dp function uses this to efficiently
    #   solve the question. The dp function can cache values as tuples of ranks are past in (both hashable)
    rks = ranks.copy()
    if 14 in ranks:
        rks.append(playingcards.Rank('A', num_value=1))
    sorted_ranks = tuple(sorted(list(set(rks)), reverse=True))
    return _straight_value_dp(sorted_ranks)


def _straight_value_dp(sorted_desc_unique_ranks: tuple) -> playingcards.Rank:
    lsr = len(sorted_desc_unique_ranks)  # Calc now, as used multiple times
    if lsr <= 4:  # Not enough cards for a straight
        return playingcards.Rank('')  # Empty rank, compares to false
    elif lsr == 5:  # Check if straight by comparing highest and lowest ranked cards (cards in unique desc order)
        if (sorted_desc_unique_ranks[0] - sorted_desc_unique_ranks[-1]) == 4:
            return sorted_desc_unique_ranks[0]
        else:
            return playingcards.Rank('')  # Empty rank, compares to false
    else:  # Check each subset, starting from the biggest cards so that we can stop early if needed
        for i in range(lsr-4):  # Using this to pull subsets of the rankgs
            sub_ranks = sorted_desc_unique_ranks[i:(i+5)]  # Grab 5 cards in a row
            if _straight_value_dp(sub_ranks):  # If we got a straight we can stop early
                return sub_ranks[0]
    return playingcards.Rank('')  # Empty rank, compares to false



def main():
    hand = texas_collections.TexasHand.from_string('8sJd')
    board = texas_collections.Board.from_string('8hTd8d8c')

    print(f'{board.filter_suit(playingcards.Suit("d", "♦"))=!s}')

    hp = TexasHandProperties(hand, board)
    print(f'{hp.quads=!s}')


if __name__ == '__main__':
    main()

