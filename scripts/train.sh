#!/usr/bin/env bash

source 00_config.sh

$PYTHON $SCRIPT \
	${DATA} \
	${DATASET} \
	${DATASET}_${PARTS} \
	${OPTS} \
	$@
