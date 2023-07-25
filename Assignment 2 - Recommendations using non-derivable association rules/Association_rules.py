from itertools import chain, combinations, filterfalse

def association_rules_avgconfidence(transactions, min_support, min_confidence):
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
    # TODO: Added
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
            support_itemset = len([t for t in transactions if itemset.issubset(t)]) / len(transactions)
            confidence = support_itemset / support_antecedent
            lift = confidence / support_itemset
            if confidence >= min_confidence:
                rules.append((antecedent, consequent, support_itemset, confidence, lift))
    rules = sorted(rules, key=lambda x: x[4], reverse=True)  # Sort by lift in descending order
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
            confidence = support_consequent / support_antecedent
            conviction = (1 - support_consequent) / (1 - confidence)
            if conviction >= min_conviction:
                rules.append((antecedent, consequent, support_consequent, confidence, conviction))
    rules = sorted(rules, key=lambda x: x[4], reverse=True)  # Sort by conviction in descending order
    return rules

def association_rules_support_confidence(transactions, min_support, min_support_confidence):
    frequent_itemsets, itemsets_by_length = apriori(transactions, min_support)
    rules = []
    for itemset in frequent_itemsets:
        for subset in filterfalse(lambda x: not x, powerset(itemset)):
            antecedent = frozenset(subset)
            consequent = itemset - antecedent
            support_antecedent = len([t for t in transactions if antecedent.issubset(t)]) / len(transactions)
            support_consequent = len([t for t in transactions if consequent.issubset(t)]) / len(transactions)
            confidence = support_consequent / support_antecedent
            support_confidence = support_consequent * confidence
            if support_confidence >= min_support_confidence:
                rules.append((antecedent, consequent, support_consequent, confidence, support_confidence))
    rules = sorted(rules, key=lambda x: x[4], reverse=True)  # Sort by support-confidence rule in descending order
    return rules