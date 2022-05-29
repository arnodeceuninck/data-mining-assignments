import pandas as pd
from tqdm import tqdm
from itertools import permutations
from more_itertools import set_partitions


def make_tid_dict(baskets):
    tid_dict = dict()
    for index, item_set in tqdm(enumerate(baskets), total=len(baskets), desc="Making tid list"):
        for item in item_set:
            frozen_set = frozenset({item})
            if frozen_set not in tid_dict:
                tid_dict[frozen_set] = set()
            tid_dict[frozen_set].add(index)
    return tid_dict


def combine_sets(tid_dict, min_support):
    new_tid_dict = dict()
    combs = list(permutations(tid_dict.items(), r=2))
    for dict_item1, dict_item2 in tqdm(combs, total=len(combs)):
        item_set1, tid_set1 = dict_item1
        item_set2, tid_set2 = dict_item2

        assert len(item_set1) == len(item_set2)
        assert len(tid_set1) >= min_support and len(tid_set2) >= min_support

        new_set = frozenset(item_set1 | item_set2)  # Same for this union operator

        # Only search for sets one size larger
        if len(new_set) != len(item_set1) + 1:
            continue

        new_set_items = tid_set1 & tid_set2  # Note: This intersection operator is new from python 3.9

        if len(new_set_items) >= min_support:
            new_tid_dict[new_set] = new_set_items
    return new_tid_dict


def filter_min_support(tid_dict, min_support):
    items = dict()
    for item in tqdm(tid_dict, total=len(tid_dict), desc="Filtering min support"):
        if len(tid_dict[item]) >= min_support:
            items[item] = tid_dict[item]
    return items


def mine_freq_itemsets(items, min_support):
    new_results = items
    freq_itemsets = items
    i = 1
    while new_results:
        i += 1
        print(f"Mining frequent itemsets of length {i}")
        new_results = combine_sets(new_results, min_support)
        freq_itemsets |= new_results
    return freq_itemsets


class Rule:

    def __init__(self, x: frozenset, y: frozenset, confidence: float, sup: float, lift: float):
        self.x = x
        self.y = y
        self.sup = sup
        self.confidence = confidence
        self.lift = lift

    def __str__(self):
        return f"{set(self.x)} -> {set(self.y)} ({self.confidence:.2f}, {self.sup})"


def get_rules(freq_itemsets, min_confidence, baskets):
    rules = list()

    for freq_itemset, tids in freq_itemsets.items():
        n_freq_itemset_transactions = len(tids)  # Note that the freq itemset is also #XandY occurances

        # Check all possible bi partitions and see if the generated rule would match the min confidence
        for x, y in set_partitions(freq_itemset, 2):
            x = frozenset(x)
            y = frozenset(y)

            if x not in freq_itemsets:
                # TODO: How come it's not possible that x is in it, but {x, y} is? This should be impossible
                continue  # x not in freq_itemset, would mean x (and thus also {x, y}) don't meet the minsup

            n_x_occurances = len(freq_itemsets[x])
            n_y_occurances = len(freq_itemsets[y])
            confidence = n_freq_itemset_transactions / n_x_occurances

            if confidence < min_confidence:
                continue

            p_xy = n_freq_itemset_transactions / len(baskets)
            p_x = n_x_occurances / len(baskets)
            p_y = n_y_occurances / len(baskets)
            lift = p_xy / (p_x * p_y)

            y = frozenset(y)
            rules.append(Rule(x=x, y=y, confidence=confidence, sup=n_freq_itemset_transactions, lift=lift))
    return rules


def get_item_nr(action_item):
    # Returns only the item id, e.g. view1234 -> returns 1234
    return int(''.join(char for char in action_item if char.isdigit()))


def get_event_type(action_item):
    # Returns only the item id, e.g. view1234 -> returns 1234
    return int(''.join(char for char in action_item if not char.isdigit()))


def all_right_set_items_new_purchases(rule):
    # Checks whether all items in the right set are purchases and that this item didn't appear with another action on the left side
    x, y = rule.x, rule.y

    x_itemids = set()
    for item in x:
        x_itemids.add(get_item_nr(item))

    for item in y:
        if not "purchase" in item or get_item_nr(item) in x_itemids:
            return False
    return True


def get_association_rules(dataset, min_support=3, min_confidence=0.7):
    dataset["action_and_product_id"] = dataset["event_type"] + dataset["product_id"].astype(str)
    baskets = dataset.groupby("user_id").action_and_product_id.apply(set).tolist()
    tid_dict = make_tid_dict(baskets)
    items = filter_min_support(tid_dict, min_support)
    freq_itemsets = mine_freq_itemsets(items, min_support)
    rules = get_rules(freq_itemsets, min_confidence, baskets)
    # filtered_rules = list(filter(lambda x: x.confidence > 0.7, rules)) # Already done in get_rules
    filtered_rules = list(filter(lambda rule: all_right_set_items_new_purchases(rule), rules))
    return filtered_rules
