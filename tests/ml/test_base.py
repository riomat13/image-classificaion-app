#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from image_app.ml.base import DogBreedClassificationLabelData


class DogBreedClassificationLabelDataTest(unittest.TestCase):

    def test_object_creaton_by_instantiation(self):
        labels = DogBreedClassificationLabelData.get_label_data()

        with self.assertRaises(RuntimeError):
            DogBreedClassificationLabelData()

        labels2 = DogBreedClassificationLabelData.get_label_data()
        self.assertTrue(labels is labels2)

    def test_label_set_creation(self):
        labels = DogBreedClassificationLabelData.get_label_data()
        self.assertEqual(len(labels), 121)

        # match order in iteration and
        # fetching by index or by label name
        for i, label in enumerate(labels):
            self.assertEqual(label, labels[i])
            self.assertEqual(i, labels[label])

    def test_label_sets(self):
        labels = DogBreedClassificationLabelData.get_label_data()

        self.assertEqual(labels.get_label_count(), 121)

        # handle nested label
        target = [0, 1, 2, 3, [4, 5], 'Affenpinscher']
        label_names = labels[target]
        self.assertEqual(len(label_names), len(target))
        self.assertEqual(len(label_names[4]), 2)
        self.assertEqual(label_names[0], 'others')
        self.assertEqual(label_names[-1], 1)

        with self.assertRaises(TypeError):
            labels[10.0]

        with self.assertRaises(TypeError):
            labels[(10, 20, 10.5, 'Affenpinscher')]


if __name__ == '__main__':
    unittest.main()
