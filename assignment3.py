from mlxtend.preprocessing import TransactionEncoder
from itertools import permutations
import pandas as pd
import numpy as np

fruits = ['Apple', 'Lemon', 'Grape', 'Banana', 'Orange', 'Kiwi', 'Pear', 'Tangerine', 'Watermelon', 'Cantaloupe', 'Lime', 'Strawberry', 'Blueberry', 'Cherry', None, None, None, None]

np.random.seed(42)
# Given transaction data
data = {
    'Fruit 1': np.random.choice(fruits, size=500),
    'Fruit 2': np.random.choice(fruits, size=500),
    'Fruit 3': np.random.choice(fruits, size=500),
    'Fruit 4': np.random.choice(fruits, size=500),
    'Fruit 5': np.random.choice(fruits, size=500),
    'Fruit 6': np.random.choice(fruits, size=500),
    'Fruit 7': np.random.choice(fruits, size=500),
    'Fruit 8': np.random.choice(fruits, size=500),
    'Fruit 9': np.random.choice(fruits, size=500),
    'Fruit 10': np.random.choice(fruits, size=500),
}

df = pd.DataFrame(data)

# Convert transaction data to a list of lists
transactions = df.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()


# Convert transactions to a one-hot encoded format
te = TransactionEncoder()
one_hot_encoded = te.fit(transactions).transform(transactions)

# Create a DataFrame from the one-hot encoded format
df = pd.DataFrame(one_hot_encoded, columns=te.columns_)


def generate_itemsets(transactions):
    """
    Generate all possible itemsets from a list of transactions.

    Parameters:
    - transactions: List of transactions, where each transaction is a set of items.

    Returns:
    - List of all possible itemsets.
    """
    all_items = set(item for transaction in transactions for item in transaction)
    all_itemsets = []

    # Generate itemsets of different sizes
    for size in range(1, 5):
        itemsets = permutations(all_items, size)
        all_itemsets.extend(list(itemset) for itemset in itemsets)

    return all_itemsets

all_itemsets = generate_itemsets(transactions)

def generate_support(data, itemsets):
  support = {}
  for transaction in data:
    for item in transaction:
      if item in support:
        support[item] += 1
      else:
        support[item] = 1

  numTransactions = len(data)

  itemSupport = {item: count / numTransactions for item, count in support.items()}

  for itemset in itemsets:
      itemsetTuple = tuple(itemset)
      itemsetCount = 0
      for transaction in data:
          if all(item in transaction for item in itemset):
                itemsetCount += 1
      itemSupport[itemsetTuple] = itemsetCount / numTransactions
  #print(itemSupport)    
  return itemSupport

#generate_support(transactions, all_itemsets)

def generate_confidence(itemset, antecedent, support):
    if isinstance(itemset, str):
        itemset = {itemset}
    if isinstance(antecedent, str):
        antecedent = {antecedent}
    support_itemset = support.get(frozenset(itemset), 0)
    support_antecedent = support.get(frozenset(antecedent), 0)

    # Avoid division by zero if the antecedent has no support
    if support_antecedent == 0:
        return 0
    confidence = support_itemset / support_antecedent

    print(f'Confidence of {antecedent} -> {itemset} = {confidence}')
    return confidence

#support_dict = generate_support(transactions, all_itemsets)

# Example call to generate_confidence
#confidence_value = generate_confidence({'Apple', 'Banana'}, {'Apple'}, support_dict)
#print(f'Returned Confidence Value: {confidence_value}')

def prune_infrequent(itemset_support, min_support):
  #Code goes here, feel free to add parameters

    pruned_itemset_support = {itemset: support for itemset, support in itemset_support.items() if support >= min_support}

    print("\nFrequent itemsets after pruning infrequent itemsets:")
    print(pruned_itemset_support)

    #print(pruned_itemset_support)
    return pruned_itemset_support

#pruned_itemset_support = prune_infrequent(itemset_support, 0.1)

def is_closed(itemset, all_itemsets, itemset_support):
  #Code goes here, feel free to add parameters
  itemset_sup = itemset_support.get(frozenset(itemset), 0)
  for other_itemset in all_itemsets:
      if set(itemset).issubset(set(other_itemset)) and itemset_support.get(frozenset(other_itemset), 0) == itemset_sup:
            return False
  return True


def is_maximal(itemset, all_itemsets, itemset_support, min_support):
  #Code goes here, feel free to add parameters
  for other_itemset in all_itemsets:
        if set(itemset).issubset(set(other_itemset)) and itemset_support.get(frozenset(other_itemset), 0) >= min_support:
            return False
  return True

# Define all_itemsets based on the keys in support_dict
#all_itemsets = list(support_dict.keys())

#result = is_maximal({'Apple', 'Banana'}, all_itemsets, support_dict, 0.1)
#print(f"Is the itemset {{'Apple', 'Banana'}} maximal? {result}")
