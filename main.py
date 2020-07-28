#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")
import logging

from svm_training.utils import parser, visualization
from svm_training.core import Trainer

from cvdatasets import AnnotationType


def main(args):

	KEY = f"{args.dataset}_{args.parts}.{args.feature_model}"
	logging.info(f"===== Setup key: {KEY} =====".format())

	annot = AnnotationType.new_annotation(args, load_strict=False)

	train, val = map(annot.new_dataset, ["train", "test"])
	logging.info("Loaded {} train and {} test images".format(len(train), len(val)))

	train_feats = train.features
	val_feats = val.features

	assert train_feats is not None and val_feats is not None, \
		"No features found!"

	logging.info("Feature shapes: {} / {}".format(train_feats.shape, val_feats.shape))

	if args.show_feature_stats:
		part_names = None
		if hasattr(annot, "part_name_list"):
			part_names = list(annot.part_name_list)
			if "NAC" in args.parts:
				part_names *= 2

			elif "GLOBAL" in args.parts:
				part_names = []

			part_names.append("GLOBAL")

		visualization.show_feature_stats(train_feats, KEY, part_names)

	trainer = Trainer.new(args, train, val, KEY)

	trainer.evaluate()


main(parser.train_args())
