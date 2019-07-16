class BaseExamplesGenerator(object):
    def split_examples(self, metadata, first_fraction):
        first_group = []
        second_group = []

        labels = set(m.label for m in metadata)
        for label in labels:
            class_metadata = [m for m in metadata if m.label == label]
            class_metadata = sorted(class_metadata, key=lambda m: m.source_id)

            split_point = int(len(class_metadata) * first_fraction)
            split_point, found = self.__find_best_split_point(class_metadata, split_point)

            if not found:
                print("Warning: couldn't find valid split, class {}, ratio {}:{2}".format(label, first_fraction, 1-first_fraction))

            first_group.extend(class_metadata[:split_point])
            second_group.extend(class_metadata[split_point:])

        return first_group, second_group

    def __find_best_split_point(self, metadata, split_point):
        def is_valid_split(metadata, split_point):
           return metadata[split_point].source_id != metadata[split_point - 1].source_id 

        if is_valid_split(metadata, split_point):
            return split_point, True
        else:
            found = False
            for distance in range(1, max(split_point, len(metadata) - split_point - 1)):
                split_point_left = split_point - distance
                if split_point_left > 0 and is_valid_split(metadata, split_point_left):
                    split_point = split_point_left
                    found = True
                    break

                split_point_right = split_point + distance
                if split_point_right < (len(metadata) - 1) and is_valid_split(metadata, split_point_right):
                    split_point = split_point_right
                    found = True
                    break

            return split_point, found

    #abstract members

    def get_examples_meta(self):
        #returns list of { Example_Metadata }
        #implement in derived class
        raise NotImplementedError()

    def get_examples(self, source_id, metadata):
        #returns list of { Example }
        #implement in derived class
        raise NotImplementedError()