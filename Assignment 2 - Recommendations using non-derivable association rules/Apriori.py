from itertools import chain, combinations, filterfalse

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def join_set(itemsets, k):
    return set(
        [itemset1.union(itemset2) for itemset1 in itemsets for itemset2 in itemsets if len(itemset1.union(itemset2)) == k]
    )

def itemsets_support(transactions, itemsets, min_support):
    support_count = {itemset: 0 for itemset in itemsets}
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                support_count[itemset] += 1
    n_transactions = len(transactions)
    return {itemset: support / n_transactions for itemset, support in support_count.items() if support / n_transactions >= min_support}

def apriori(transactions, min_support):
    items = set(chain(*transactions))
    itemsets = [frozenset([item]) for item in items]
    # The empty set is removed
    itemsets_by_length = []
    k = 1
    while itemsets:
        support_count = itemsets_support(transactions, itemsets, min_support)
        # if the support_count set is empty then not necessary to apppend
        if support_count:
            itemsets_by_length.append(set(support_count.keys()))
        k += 1
        itemsets = join_set(itemsets, k)
    frequent_itemsets = set(chain(*itemsets_by_length))
    return frequent_itemsets, itemsets_by_length

def recommend_items(input_items, rules, top_n=5):
    recommendations = {}
    for antecedent, consequent, support, confidence in rules:
        if antecedent.issubset(input_items) and not consequent.issubset(input_items):
            for item in consequent:
                if item not in input_items:
                    if item not in recommendations:
                        recommendations[item] = []
                    recommendations[item].append((confidence, support))
    recommendations = {
        item: (sum(conf for conf, _ in item_rules) / len(item_rules), sum(sup for _, sup in item_rules) / len(item_rules))
        for item, item_rules in recommendations.items()
    }
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: (-x[1][0], -x[1][1]))
    return [item for item, _ in sorted_recommendations[:top_n]]


def association_rules_average_confidence(transactions, min_support, min_confidence):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, confidence))
    rules = sorted(rules, key=lambda x: x[3], reverse=True)  # Sort by confidence in descending order
    return rules

def association_rules_laplace(transactions, min_support, min_confidence):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            laplace = (support_itemset + 1) / (support_antecedent + 2)
            if laplace >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, laplace))
    rules = sorted(rules, key=lambda x: x[3], reverse=True)  # Sort by confidence in descending order
    return rules

def association_rules_lift(transactions, min_support, min_confidence):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_consequent = len([t for t in transactions if consequent.issubset(t)]) / len(transactions)
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            lift = confidence / support_consequent
            if lift > 1:
                rules.append((antecedent, consequent, support_itemset, lift))
    rules = sorted(rules, key=lambda x: x[3], reverse=True)  # Sort by lift in descending order
    return rules

def association_rules_conviction(transactions, min_support, min_conviction):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_consequent = len([t for t in transactions if consequent.issubset(t)]) / len(transactions)
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            if confidence < 1:
                conviction = (1 - support_consequent) / (1 - confidence)
            else:
                conviction = float('inf')  # or assign a large positive value
            if conviction > 1:
                rules.append((antecedent, consequent, support_consequent, conviction))
    rules = sorted(rules, key=lambda x: x[3], reverse=True)  # Sort by conviction in descending order
    return rules

# Example usage
input_items = {"A", "B"}

transactions = [
  {"A", "B", "C"},
  {"A", "B"},
  {"A", "C"},
  {"A"},
  {"B", "C"},
  {"B"},
  {"C"},
]
min_support = 0.2
min_confidence = 0.5

print("##################")
print("AVERAGE CONFIDENCE")
print("##################")
rules = association_rules_average_confidence(transactions, min_support, min_confidence)
for antecedent, consequent, support, confidence in rules:
    print(f"{antecedent} => {consequent} (support={support:.2f}, confidence={confidence:.2f})")
recommended_items = recommend_items(input_items, rules)
print("Recommended items:", recommended_items)

print("##################")
print("LAPLACE")
print("##################")
rules = association_rules_laplace(transactions, min_support, min_confidence)
for antecedent, consequent, support, laplace in rules:
    print(f"{antecedent} => {consequent} (support={support:.2f}, laplace={laplace:.2f})")
recommended_items = recommend_items(input_items, rules)
print("Recommended items:", recommended_items)

print("##################")
print("CONVICTION")
print("##################")
rules = association_rules_conviction(transactions, min_support, min_confidence)
for antecedent, consequent, support, conviction in rules:
    print(f"{antecedent} => {consequent} (support={support:.2f}, conviction={conviction:.2f})")
recommended_items = recommend_items(input_items, rules)
print("Recommended items:", recommended_items)

print("##################")
print("LIFT")
print("##################")
rules = association_rules_lift(transactions, min_support, min_confidence)
for antecedent, consequent, support, lift in rules:
    print(f"{antecedent} => {consequent} (support={support:.2f}, lift={lift:.2f})")
recommended_items = recommend_items(input_items, rules)
print("Recommended items:", recommended_items)

    
