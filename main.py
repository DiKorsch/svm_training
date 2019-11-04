#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")
import logging

from svm_training.utils import parser, visualization
from svm_training.core import Trainer

from cvdatasets.annotations import AnnotationType


def main(args):

	KEY = "{}.{}".format(args.parts, args.model_type)
	logging.info("===== Setup key: {} =====".format(KEY))

	annot_cls = AnnotationType.get(args.dataset).value

	annot = annot_cls(
		root_or_infofile=args.data,
		parts=args.parts,
		feature_model=args.model_type)

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
