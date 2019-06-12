from utils.bitmask import to_bitmask
from utils.helpers import flatten_list

class Combinator(object):
    def __init__(self, accuracy=0.005):
        self.accuracy = accuracy

    def split(self, groups, split_ratio):
        """Split groups into k subgroups with given split_ratio
        Args:
            elements: list of tuples, first el is "name", second is "count" e.g [("418", 101), ("419", 3), ...]
            split_ratio: k-length list of float, k is the number of subgroups, k[i] is relative size
            of the i-th subgroup, e.g [0.2, 0.2, 0.2, 0.2, 0.2] for 5-fold cross validation
        Returns:
            k-length list of lists of group names, e.f [["418", "419"], ["500"], ...]
        """
        splits = []

        remaining_groups = groups.copy()

        """if groups length is more then, say, 20, reduce groups number
        by combining several small groups into a bigger one
        see __decouple_groups method comments
        """
        length_limit = 19

        if len(groups) > length_limit:
            print("Combinator: groups length is", len(groups), ", reducing groups number to", length_limit)

            remaining_groups.sort(key=lambda tup: tup[1], reverse=True) 
            for i in range(length_limit, len(groups)):
                idx = (length_limit - 1) - (i % length_limit)
                name = ",".join([remaining_groups[idx][0], remaining_groups[i][0]])
                quantity = remaining_groups[idx][1] + remaining_groups[i][1]
                remaining_groups[idx] = (name, quantity)

            remaining_groups = remaining_groups[:length_limit]
        
        for i in range(len(split_ratio) - 1):
            #relative to remaining groups
            ratio = split_ratio[i] / sum(split_ratio[i:])

            subgroup = self._get_subgroup(remaining_groups, ratio)
            
            remaining_groups = [el for el in remaining_groups if not (el in subgroup)]

            splits.append(self.__decouple_groups(subgroup, groups))

        splits.append(self.__decouple_groups(remaining_groups, groups))

        return splits

    def _get_subgroup(self, elements, ratio):
        """Picks subgroup of elements taking into account given ratio
        Args:
            elements: list of tuples, first el is "name", second is "count" e.g [("418", 101), ("419", 3), ...]
            ratio: float, relative size of subgroup, e.g 0.2
        Returns:
            list of indexes of elemets in subgroup, e.g [0, 10, 11]
        """
        
        bits = len(elements)
        total_size = sum(el[1] for el in elements)

        best_combination = [0] * bits
        best_combination_error = 1.0

        for i in range(1, 2 ** bits):
            combination = to_bitmask(i, bits)
            subgroup_size = self.__subgroup_size(elements, combination)
            curr_error = abs(subgroup_size / total_size - ratio)
            if curr_error < best_combination_error:
                best_combination_error = curr_error
                best_combination = combination.copy()
                
        return [el for el, include in zip(elements, best_combination) if include == 1]

    def __subgroup_size(self, elements, mask):
        return sum(el[1] for el, include in zip(elements, mask) if include == 1)

    def __decouple_groups(self, subgroup, groups):
        """
        when original number of groups is large enough, for performance purposes
        smaller groups are combined fith large ones, for example:
            before: ("100", 200), ("101", 300), ("102", 3), ("104", 4)
            after: ("100,104", 204), ("101,102", 303)
        this method is used to split combined groups into origninal groups.
        in order to keep orginal group quantity number, original collection of groups needs to be provided
        Args:
            subgroup: combined group (ex: after)
            grops: original list of all groups
        Returns:
            ex: before
        """
        result = []

        for s in subgroup:
            for name in s[0].split(","):
                quantity = [g[1] for g in groups if g[0] == name][0]
                result.append((name, quantity))
        
        return result